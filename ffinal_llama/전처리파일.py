import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from datasets import load_dataset, Dataset
import pandas as pd
from pathlib import Path
from datetime import datetime

class BBQSDUSingleAxisPreprocessor:
    def __init__(self, target_category: str, seed: int = 42, verbose: bool = False):
        self.target_category = target_category
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        
        self.supported_categories = [
            'Race_ethnicity', 'SES', 'Age', 'Gender_identity'
        ]
        
        if target_category not in self.supported_categories:
            raise ValueError(f"Unsupported category: {target_category}")
        
        self.raw_data = None
        self.filtered_data = None
        self.train_data = None
        self.test_data = None
        self.forget_set = None
        self.retain_set = None
        self.eval_data = None
        
        self.stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'neutral_samples_excluded': 0,
            'final_training_samples': 0
        }
        
        self.processing_log = []
    
    def _log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.processing_log.append(log_entry)
        
        if self.verbose:
            print(log_entry)
    
    def load_bbq_dataset(self) -> Dataset:
        if not self.verbose:
            print(f"📥 Loading {self.target_category}...")
        
        try:
            category_mapping = {
                'Race_ethnicity': 'race_ethnicity',
                'SES': 'ses', 
                'Age': 'age',
                'Gender_identity': 'gender_identity'
            }
            
            actual_split = category_mapping.get(self.target_category)
            dataset = load_dataset("Elfsong/BBQ", split=actual_split)
            
            self._log(f"Loaded {len(dataset)} samples for {self.target_category}")
            if len(dataset) > 0:
                sample = dataset[0]
                self._log(f"Sample fields: {list(sample.keys())}")
            
            self.raw_data = dataset
            self.filtered_data = dataset
            self.stats['total_samples'] = len(dataset)
            self.stats['filtered_samples'] = len(dataset)
            
            return dataset
            
        except Exception as e:
            self._log(f"Error loading dataset: {e}", "ERROR")
            raise

    def filter_target_category(self) -> Dataset:
        self._log(f"Category already filtered: {len(self.filtered_data)} samples")
        return self.filtered_data
    
    def split_train_test(self, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        if not self.verbose:
            print(f"✂️  Splitting data...")
            
        split_data = self.filtered_data.train_test_split(
            test_size=test_size, 
            seed=self.seed
        )
        
        train_data = split_data['train']
        test_data = split_data['test']
        
        self._log(f"Split complete: Train={len(train_data)}, Test={len(test_data)}")
        
        self.train_data = train_data
        self.test_data = test_data
        
        return train_data, test_data
    
    def create_forget_retain_sets(self) -> Tuple[List[Dict], List[Dict]]:
        if not self.verbose:
            print(f"🔄 Creating forget/retain sets...")
        
        forget_set = []
        retain_set = []
        neutral_count = 0
        invalid_count=0
    
        
        for sample in self.train_data:
            if sample['target_label'] == sample['answer_label']:
                neutral_count += 1
                continue
            
            if sample['answer_label'] < 0 or sample['target_label'] < 0:
                invalid_count += 1
                continue

            ans_key_answer = f'ans{sample["answer_label"]}'
            ans_key_target = f'ans{sample["target_label"]}'

            if ans_key_answer not in sample or ans_key_target not in sample:
                invalid_count += 1
                continue
            
            context = sample['context']
            question = sample['question']
            input_prompt = f"{context}\n\nQuestion: {question}"
            
            retain_answer = sample[ans_key_answer]  
            forget_answer = sample[ans_key_target]  
            
            forget_formatted = f"<s>[INST] {input_prompt} [/INST] {forget_answer}</s>"
            retain_formatted = f"<s>[INST] {input_prompt} [/INST] {retain_answer}</s>"
            
            metadata = sample['additional_metadata']
            stereotyped_groups = metadata.get('known_stereotyped_groups', 'unknown')
            
            common_meta = {
                'category': sample['category'],
                'context_condition': sample['context_condition'],
                'question_polarity': sample.get('question_polarity', 'unknown'),
                'stereotyped_groups': stereotyped_groups,
                'social_values': metadata.get('relevant_social_values', ''),
                'example_id': sample.get('example_id', -1),
                'answer_label': sample['answer_label'],
                'target_label': sample['target_label']
            }
            
            forget_set.append({
                'text': forget_formatted,
                'input': input_prompt,
                'output': forget_answer,
                'is_biased': True,
                **common_meta
            })
            
            retain_set.append({
                'text': retain_formatted,
                'input': input_prompt,
                'output': retain_answer,
                'is_biased': False,
                **common_meta
            })
        
        self.stats['neutral_samples_excluded'] = neutral_count
        self.stats['final_training_samples'] = len(forget_set)
        
        self._log(f"Created {len(forget_set)} forget/retain pairs, excluded {neutral_count} neutral samples")
        
        self.forget_set = forget_set
        self.retain_set = retain_set
        
        return forget_set, retain_set
    
    def create_evaluation_set(self, eval_size: int = 2500) -> Dataset:
        available_samples = len(self.test_data)
        actual_eval_size = min(eval_size, available_samples)
        
        if available_samples < eval_size:
            eval_data = self.test_data
        else:
            eval_data = self.test_data.shuffle(seed=self.seed).select(range(actual_eval_size))
        
        self._log(f"Created evaluation set: {len(eval_data)} samples")
        self.eval_data = eval_data
        return eval_data
    
    def validate_preprocessing(self) -> bool:
        if not self.verbose:
            print(f"✅ Validating...")
            
        try:
            assert self.forget_set is not None and len(self.forget_set) > 0
            assert len(self.forget_set) == len(self.retain_set)
            
            for sample in self.forget_set:
                assert sample['answer_label'] != sample['target_label']
            
            sample_forget = self.forget_set[0]
            assert sample_forget['text'].startswith('<s>[INST]')
            assert '[/INST]' in sample_forget['text']
            assert sample_forget['text'].endswith('</s>')
            
            self._log("All validation checks passed!")
            return True
            
        except Exception as e:
            self._log(f"Validation failed: {e}", "ERROR")
            return False
    
    def save_processed_data(self, base_output_dir: str = "./processed_bbq") -> str:
        output_dir = Path(base_output_dir) / f"{self.target_category}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.verbose:
            print(f"💾 Saving to {output_dir}...")
        
        with open(output_dir / "forget_set.json", 'w', encoding='utf-8') as f:
            json.dump(self.forget_set, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "retain_set.json", 'w', encoding='utf-8') as f:
            json.dump(self.retain_set, f, ensure_ascii=False, indent=2)
        
        if self.eval_data:
            eval_dict = [dict(sample) for sample in self.eval_data]
            with open(output_dir / "eval_data.json", 'w', encoding='utf-8') as f:
                json.dump(eval_dict, f, ensure_ascii=False, indent=2)
        
        detailed_info = self._generate_detailed_info()
        with open(output_dir / "detailed_analysis.txt", 'w', encoding='utf-8') as f:
            f.write(detailed_info)
        
        with open(output_dir / "processing_log.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(self.processing_log))
        
        metadata = {
            'target_category': self.target_category,
            'dataset_source': 'Elfsong/BBQ',
            'statistics': self.stats,
            'seed': self.seed,
            'preprocessing_date': pd.Timestamp.now().isoformat(),
            'data_quality': {
                'neutral_samples_excluded': True,
                'single_category_focus': True,
                'mistral_chat_format': True
            }
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        stats_df = pd.DataFrame([{
            'category': self.target_category,
            'total_samples': self.stats['total_samples'],
            'filtered_samples': self.stats['filtered_samples'],
            'neutral_excluded': self.stats['neutral_samples_excluded'],
            'final_training': self.stats['final_training_samples'],
            'evaluation': len(self.eval_data) if self.eval_data else 0
        }])
        stats_df.to_csv(output_dir / "statistics.csv", index=False)
        
        self._log(f"All files saved to {output_dir}")
        return str(output_dir)
    
    def _generate_detailed_info(self) -> str:
        info = []
        info.append(f"DETAILED ANALYSIS - {self.target_category}")
        info.append("=" * 60)
        
        info.append(f"\nSTATISTICS:")
        info.append(f"Original samples: {self.stats['total_samples']:,}")
        info.append(f"Final training: {self.stats['final_training_samples']:,}")
        info.append(f"Neutral excluded: {self.stats['neutral_samples_excluded']:,}")
        info.append(f"Evaluation: {len(self.eval_data) if self.eval_data else 0:,}")
        
        if self.stats['filtered_samples'] > 0:
            exclusion_rate = (self.stats['neutral_samples_excluded'] / 
                            self.stats['filtered_samples']) * 100
            info.append(f"Exclusion rate: {exclusion_rate:.1f}%")
        
        if self.forget_set:
            context_dist = Counter([s['context_condition'] for s in self.forget_set])
            polarity_dist = Counter([s['question_polarity'] for s in self.forget_set])
            
            info.append(f"\nContext conditions: {dict(context_dist)}")
            info.append(f"Question polarity: {dict(polarity_dist)}")
        
        if self.forget_set:
            info.append(f"\nSAMPLE EXAMPLES:")
            info.append("-" * 40)
            
            forget_sample = self.forget_set[0]
            info.append("FORGET SET (Biased Response):")
            info.append(f"Context: {forget_sample['context_condition']}")
            info.append(f"Input: {forget_sample['input'][:200]}...")
            info.append(f"Output: {forget_sample['output']}")
            info.append(f"Labels: answer={forget_sample['answer_label']}, target={forget_sample['target_label']}")
            
            info.append("")
            
            retain_sample = self.retain_set[0]
            info.append("RETAIN SET (Fair Response):")
            info.append(f"Input: {retain_sample['input'][:200]}...")
            info.append(f"Output: {retain_sample['output']}")
        
        return "\n".join(info)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        if not self.verbose:
            print(f"🚀 Processing {self.target_category}...")
        
        try:
            self.load_bbq_dataset()
            self.filter_target_category()
            
            if len(self.filtered_data) == 0:
                print(f"❌ No data for {self.target_category}")
                return None
            
            self.split_train_test()
            self.create_forget_retain_sets()
            self.create_evaluation_set()
            
            if not self.validate_preprocessing():
                print(f"❌ Validation failed for {self.target_category}")
                return None
            
            if not self.verbose:
                print(f"✅ {self.target_category} completed!")
            
            return {
                'category': self.target_category,
                'forget_set': self.forget_set,
                'retain_set': self.retain_set,
                'eval_data': self.eval_data,
                'statistics': self.stats
            }
            
        except Exception as e:
            print(f"❌ {self.target_category} failed: {str(e)[:100]}...")
            self._log(f"Pipeline failed: {e}", "ERROR")
            return None


def process_all_categories(
    categories: List[str] = None, 
    base_output_dir: str = "./processed_bbq",
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    if categories is None:
        categories = ['Race_ethnicity', 'SES', 'Age', 'Gender_identity']
    
    print("🌟 BBQ-SDU PREPROCESSING")
    print(f"📁 Output: {base_output_dir}")
    print(f"🎯 Categories: {len(categories)}")
    print("=" * 50)
    
    results = {}
    successful_categories = []
    failed_categories = []
    
    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] {category}")
        
        try:
            preprocessor = BBQSDUSingleAxisPreprocessor(
                target_category=category, 
                seed=seed,
                verbose=verbose
            )
            
            result = preprocessor.run_complete_pipeline()
            
            if result is not None:
                output_path = preprocessor.save_processed_data(base_output_dir)
                result['output_path'] = output_path
                results[category] = result
                successful_categories.append(category)
            else:
                failed_categories.append(category)
                
        except Exception as e:
            print(f"❌ Failed: {str(e)[:50]}...")
            failed_categories.append(category)
    
    print(f"\n🎯 SUMMARY")
    print("=" * 30)
    print(f"✅ Successful: {len(successful_categories)}")
    print(f"❌ Failed: {len(failed_categories)}")
    
    if successful_categories:
        print(f"📊 Categories: {successful_categories}")
        
        summary_stats = []
        for category in successful_categories:
            if category in results:
                stats = results[category]['statistics']
                stats['category'] = category
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_path = Path(base_output_dir) / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"📄 Summary saved: {summary_path}")
    
    return {
        'successful_categories': successful_categories,
        'failed_categories': failed_categories,
        'results': results,
        'base_output_dir': base_output_dir
    }


def main():
    results = process_all_categories(
        categories=['Race_ethnicity', 'SES', 'Age', 'Gender_identity'],
        base_output_dir="./processed_bbq",
        seed=42,
        verbose=False
    )
    
    print(f"\n🚀 Processing complete!")
    print(f"📁 Data: {results['base_output_dir']}")
    print(f"📊 Success rate: {len(results['successful_categories'])}/{len(results['successful_categories']) + len(results['failed_categories'])}")
    
    return results


def test_single_category_verbose(category: str = "Age"):
    preprocessor = BBQSDUSingleAxisPreprocessor(target_category=category, verbose=True)
    result = preprocessor.run_complete_pipeline()
    if result:
        preprocessor.save_processed_data("./test_output")
    return result


if __name__ == "__main__":
    main()