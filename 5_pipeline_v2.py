import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset
import logging
import gc
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AggressiveSDUConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_length: int = 256
    batch_size: int = 4  
    num_train_epochs: int = 10  
    learning_rates: List[float] = None
    seeds: List[int] = None
    lora_r: int = 64  
    lora_alpha: int = 128  
    lora_target_modules: List[str] = None
    sign_merge_threshold: float = 0.5  
    projection_scaling: float = 5.0  
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-3, 5e-3] 
        if self.seeds is None:
            self.seeds = [42, 123] 
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

class AggressiveBBQSDUTrainer:
    
    def __init__(self, config: AggressiveSDUConfig, base_data_dir: str = "./processed_bbq"):
        self.config = config
        self.base_data_dir = Path(base_data_dir)
        self.categories = ['Race_ethnicity', 'SES', 'Gender_identity']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.task_vectors = {}
        self.bias_vectors = {}
        
        logger.info(f"Aggressive trainer initialized on device: {self.device}")
        logger.info(f"Config: lr={self.config.learning_rates}, epochs={self.config.num_train_epochs}")
        logger.info(f"LoRA: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
    
    def load_model_for_training(self):
        logger.info(f"Loading model for aggressive training: {self.config.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=None,
            load_in_8bit=False,
            trust_remote_code=True
        )
        
        model = model.to(self.device)
        return model
    
    def load_model_for_inference(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=False,
            trust_remote_code=True
        )
        return model
    
    def create_aggressive_lora_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,  
            lora_alpha=self.config.lora_alpha,  
            lora_dropout=0.05,  
            target_modules=self.config.lora_target_modules,  
            bias="none"
        )
    
    def prepare_dataset(self, data_path: str, max_samples: int = None) -> Dataset:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples is not None:
            data = data[:max_samples]
            logger.info(f"Limited dataset to {len(data)} samples")
        
        texts = [item['text'] for item in data]
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def extract_lora_delta_weights(self, peft_model) -> Dict[str, torch.Tensor]:
        delta_weights = {}
        
        logger.info("Extracting LoRA delta weights with aggressive scaling...")
        
        peft_state_dict = peft_model.state_dict()
        
        lora_pairs = {}
        for name, param in peft_state_dict.items():
            if 'lora_A' in name:
                base_name = name.replace('.lora_A.default.weight', '').replace('.lora_A.weight', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['A'] = param
            elif 'lora_B' in name:
                base_name = name.replace('.lora_B.default.weight', '').replace('.lora_B.weight', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['B'] = param
        
        logger.info(f"Found {len(lora_pairs)} LoRA pairs")
        
        total_norm_before = 0.0
        total_norm_after = 0.0
        
        for base_name, matrices in lora_pairs.items():
            if 'A' in matrices and 'B' in matrices:
                lora_A = matrices['A']
                lora_B = matrices['B']
                
                scaling = self.config.lora_alpha / self.config.lora_r
                
                delta_weight = (lora_B @ lora_A) * scaling
                
                norm_before = torch.norm(delta_weight).item()
                total_norm_before += norm_before
                
                delta_weight = delta_weight * 2.0  
                
                norm_after = torch.norm(delta_weight).item()
                total_norm_after += norm_after
                
                clean_name = base_name.replace('base_model.model.', '')
                if not clean_name.endswith('.weight'):
                    clean_name += '.weight'
                
                delta_weights[clean_name] = delta_weight.cpu().float()
                
                logger.debug(f"Extracted {clean_name}: norm {norm_before:.6f} -> {norm_after:.6f}")
        
        logger.info(f"Extracted {len(delta_weights)} LoRA delta weights")
        logger.info(f"Total norm amplification: {total_norm_before:.6f} -> {total_norm_after:.6f}")
        
        return delta_weights
    
    def apply_aggressive_bias_removal(self, bias_vector: Dict[str, torch.Tensor], 
                                    output_model_name: str = "debiased_model") -> object:
        logger.info(f"Applying AGGRESSIVE bias removal: {output_model_name}")
        logger.info(f"Projection scaling factor: {self.config.projection_scaling}")
        
        if not bias_vector:
            logger.error("Empty bias vector, cannot apply bias removal")
            return None
        
        model = self.load_model_for_inference()
        
        with torch.no_grad():
            applied_count = 0
            total_count = len(bias_vector)
            total_projection_magnitude = 0.0
            
            for param_name, bias_delta in bias_vector.items():
                param_dict = dict(model.named_parameters())
                
                if param_name in param_dict:
                    original_param = param_dict[param_name]
                    
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        param_device = original_param.device
                    else:
                        param_device = next(model.parameters()).device
                    
                    bias_delta_gpu = bias_delta.to(param_device, dtype=original_param.dtype)
                    
                    bias_delta_gpu = bias_delta_gpu * self.config.projection_scaling
                    
                    if original_param.shape == bias_delta_gpu.shape:
                        orig_flat = original_param.view(-1)
                        bias_flat = bias_delta_gpu.view(-1)
                        
                        inner_product = torch.dot(orig_flat, bias_flat)
                        bias_norm_sq = torch.dot(bias_flat, bias_flat)
                        
                        if bias_norm_sq > 1e-8:
                            # Projection: proj_v(u) = <u,v>/<v,v> * v
                            projection_coef = inner_product / bias_norm_sq
                            projection = projection_coef * bias_delta_gpu
                            
                            original_param.data -= projection
                            applied_count += 1
                            
                            proj_magnitude = torch.norm(projection).item()
                            total_projection_magnitude += proj_magnitude
                            
                            logger.debug(f"AGGRESSIVE projection to {param_name}: "
                                       f"coef={projection_coef:.6f}, magnitude={proj_magnitude:.6f}")
                        else:
                            logger.warning(f"Bias norm too small for {param_name}, skipping")
                    else:
                        logger.warning(f"Shape mismatch for {param_name}")
                else:
                    logger.warning(f"Parameter {param_name} not found in model")
        
        logger.info(f"Applied AGGRESSIVE bias removal to {applied_count}/{total_count} parameters")
        logger.info(f"Total projection magnitude: {total_projection_magnitude:.6f}")
        
        if applied_count == 0:
            logger.error("No parameters were modified during bias removal!")
            return None
        
        output_path = f"./aggressive_debiased_models/{output_model_name}"
        os.makedirs(output_path, exist_ok=True)
        
        try:
            model.save_pretrained(output_path, safe_serialization=True)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Aggressively debiased model saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
        
        return model
    
    def calculate_simple_bias_score(self, model, data_path: str, max_samples: int = 100) -> Dict[str, float]:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return {'bias_score': 0.5, 'stereotype_rate': 0.5, 'anti_stereotype_rate': 0.5}

        model.eval()

        total_samples = 0
        stereotypical_predictions = 0
        anti_stereotypical_predictions = 0
        correct_predictions = 0

        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device = list(model.hf_device_map.values())[0]
        else:
            device = next(model.parameters()).device

        with torch.no_grad():
            for i, item in enumerate(data[:max_samples]):
                try:
                    required_fields = ['context', 'question', 'ans0', 'ans1', 'ans2']
                    if not all(field in item for field in required_fields):
                        continue

                    context = item['context']
                    question = item['question']
                    answers = [item['ans0'], item['ans1'], item['ans2']]

                    answer_label = item.get('answer_label', -1)
                    target_label = item.get('target_label', -1)

                    if answer_label < 0 or answer_label >= len(answers):
                        continue
                    if target_label < 0 or target_label >= len(answers):
                        continue

                    total_samples += 1

                    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
                    
                    answer_probs = self._calculate_simple_probabilities(
                        model, prompt, answers, device
                    )

                    if not answer_probs:
                        continue

                    predicted_idx = np.argmax(answer_probs)

                    if predicted_idx == answer_label:
                        correct_predictions += 1

                    if predicted_idx == target_label:
                        stereotypical_predictions += 1
                    else:
                        anti_stereotypical_predictions += 1

                except Exception as e:
                    logger.debug(f"Error processing sample {i}: {e}")
                    continue

        if total_samples == 0:
            logger.warning("No valid samples for bias calculation")
            return {'bias_score': 0.5, 'stereotype_rate': 0.5, 'anti_stereotype_rate': 0.5}

        accuracy = correct_predictions / total_samples
        stereotype_rate = stereotypical_predictions / total_samples
        anti_stereotype_rate = anti_stereotypical_predictions / total_samples

        bias_score = stereotype_rate

        balance_score = abs(0.5 - stereotype_rate)

        results = {
            'bias_score': bias_score,
            'stereotype_rate': stereotype_rate,
            'anti_stereotype_rate': anti_stereotype_rate,
            'accuracy': accuracy,
            'balance_score': 1.0 - balance_score,  
            'total_samples': total_samples
        }

        logger.info(f"Bias calculation results:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Stereotype rate: {stereotype_rate:.3f}")
        logger.info(f"  Anti-stereotype rate: {anti_stereotype_rate:.3f}")
        logger.info(f"  Bias score: {bias_score:.3f}")

        return results

    def _calculate_simple_probabilities(self, model, prompt: str, answers: List[str], device) -> List[float]:
        answer_probs = []

        try:
            for answer in answers:
                try:
                    full_text = prompt + f" {answer.strip()}"

                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_length
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss

                    prob = torch.exp(-loss).item()
                    answer_probs.append(prob)

                except Exception as e:
                    logger.debug(f"Error calculating probability for answer '{answer}': {e}")
                    answer_probs.append(0.0)

            if sum(answer_probs) > 0:
                total_prob = sum(answer_probs)
                answer_probs = [p / total_prob for p in answer_probs]
            else:
                answer_probs = [1.0/len(answers)] * len(answers)

        except Exception as e:
            logger.debug(f"Error in probability calculation: {e}")
            return [1.0/len(answers)] * len(answers)

        return answer_probs

    def calculate_perplexity(self, model, data_path: str, max_samples: int = 50) -> float:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return float('inf')
        
        model.eval()
        total_loss = 0
        total_count = 0
        
        with torch.no_grad():
            for i, item in enumerate(data[:max_samples]):
                try:
                    text = item['text']
                    
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True,
                        max_length=self.config.max_length
                    )
                    
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        device = list(model.hf_device_map.values())[0]
                    else:
                        device = next(model.parameters()).device
                    
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        total_count += 1
                        
                except Exception as e:
                    logger.warning(f"Perplexity calculation error for sample {i}: {e}")
                    continue
        
        if total_count == 0:
            logger.warning("No valid samples for perplexity calculation")
            return float('inf')
        
        avg_loss = total_loss / total_count
        perplexity = np.exp(avg_loss)
        
        return perplexity

    def train_single_run(self, category: str, lr: float, seed: int, max_samples: int = None) -> str:
        run_name = f"{category.lower()}_aggressive_lr{lr}_seed{seed}"
        output_dir = f"./aggressive_checkpoints/{run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting AGGRESSIVE training: {run_name}")
        logger.info(f"LR: {lr}, Epochs: {self.config.num_train_epochs}, LoRA r/alpha: {self.config.lora_r}/{self.config.lora_alpha}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            forget_data_path = self.base_data_dir / category / "forget_set.json"
            if not forget_data_path.exists():
                logger.error(f"Data file not found: {forget_data_path}")
                return ""
                
            dataset = self.prepare_dataset(str(forget_data_path), max_samples)
            
            if len(dataset) == 0:
                logger.error("Empty dataset")
                return ""
            
            model = self.load_model_for_training()
            
            lora_config = self.create_aggressive_lora_config()
            model = get_peft_model(model, lora_config)
            
            model.train()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info(f"Trainable params: {trainable_params:,} / Total: {total_params:,} "
                       f"({100*trainable_params/total_params:.2f}%)")
            
            if trainable_params == 0:
                logger.error("No trainable parameters found!")
                return ""
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.num_train_epochs,  
                per_device_train_batch_size=self.config.batch_size,  
                learning_rate=lr,  
                fp16=True,
                save_strategy="no",
                dataloader_drop_last=True,
                gradient_checkpointing=False,
                dataloader_pin_memory=False,
                seed=seed,
                remove_unused_columns=False,
                report_to=None,
                warmup_steps=0,
                weight_decay=0.0,  
                gradient_accumulation_steps=1,
                logging_steps=5,  
                dataloader_num_workers=0,
                max_grad_norm=1.0,  
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            
            start_time = time.time()
            logger.info("Starting AGGRESSIVE training process...")
            
            trainer.train()
            training_time = time.time() - start_time
            logger.info(f"AGGRESSIVE training completed in {training_time:.2f} seconds")
            
            delta_weights = self.extract_lora_delta_weights(model)
            
            if not delta_weights:
                logger.error("No delta weights extracted!")
                return ""
            
            torch.save(delta_weights, f"{output_dir}/task_vector.pt")
            
            metadata = {
                'category': category,
                'learning_rate': lr,
                'seed': seed,
                'training_time': training_time,
                'dataset_size': len(dataset),
                'num_parameters': len(delta_weights),
                'aggressive_mode': True,
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha
            }
            with open(f"{output_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"AGGRESSIVE training completed successfully: {run_name}")
            
        except Exception as e:
            logger.error(f"AGGRESSIVE training failed for {run_name}: {str(e)}")
            return ""
        
        finally:
            try:
                del model, trainer
            except:
                pass
            torch.cuda.empty_cache()
            gc.collect()
        
        return output_dir

    def train_category_all_runs(self, category: str, max_samples: int = None) -> List[str]:
        logger.info(f"Training all AGGRESSIVE runs for category: {category}")
        
        output_dirs = []
        total_runs = len(self.config.learning_rates) * len(self.config.seeds)
        current_run = 0
        
        for lr in self.config.learning_rates:
            for seed in self.config.seeds:
                current_run += 1
                logger.info(f"AGGRESSIVE Run {current_run}/{total_runs}: lr={lr}, seed={seed}")
                
                output_dir = self.train_single_run(category, lr, seed, max_samples)
                if output_dir:
                    output_dirs.append(output_dir)
                    logger.info(f"✅ Successful AGGRESSIVE run: {lr}, {seed}")
                else:
                    logger.warning(f"❌ Failed AGGRESSIVE run: {category}, lr={lr}, seed={seed}")
        
        logger.info(f"Completed {len(output_dirs)}/{total_runs} AGGRESSIVE runs for {category}")
        return output_dirs

    def load_aggressive_task_vectors(self, category: str) -> List[Dict[str, torch.Tensor]]:
        task_vectors = []
        
        for lr in self.config.learning_rates:
            for seed in self.config.seeds:
                run_name = f"{category.lower()}_aggressive_lr{lr}_seed{seed}"
                vector_path = f"./aggressive_checkpoints/{run_name}/task_vector.pt"
                
                if os.path.exists(vector_path):
                    try:
                        task_vector = torch.load(vector_path, map_location='cpu')
                        if task_vector:
                            task_vectors.append(task_vector)
                            logger.debug(f"Loaded aggressive task vector: {run_name}")
                        else:
                            logger.warning(f"Empty aggressive task vector: {run_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {vector_path}: {e}")
                else:
                    logger.warning(f"Aggressive task vector not found: {vector_path}")
        
        logger.info(f"Loaded {len(task_vectors)} valid aggressive task vectors for {category}")
        return task_vectors

    def sign_merge_vectors(self, task_vectors: List[Dict[str, torch.Tensor]], 
                          threshold: float = None) -> Dict[str, torch.Tensor]:
        if not task_vectors:
            logger.warning("No task vectors to merge")
            return {}
        
        if len(task_vectors) == 1:
            logger.info("Only one vector, returning as-is")
            return task_vectors[0]
        
        if threshold is None:
            threshold = self.config.sign_merge_threshold  
            
        logger.info(f"Sign-merging {len(task_vectors)} vectors with threshold {threshold}")
        
        param_names = set(task_vectors[0].keys())
        for vector in task_vectors[1:]:
            param_names = param_names.intersection(set(vector.keys()))
        
        if not param_names:
            logger.error("No common parameters found across vectors")
            return {}
        
        merged_vector = {}
        total_params = 0
        kept_params = 0
        
        for param_name in param_names:
            try:
                param_tensors = []
                for vector in task_vectors:
                    if param_name in vector:
                        param_tensors.append(vector[param_name])
                
                if len(param_tensors) != len(task_vectors):
                    logger.warning(f"Inconsistent parameter {param_name}, skipping")
                    continue
                
                stacked = torch.stack(param_tensors)
                
                signs = torch.sign(stacked)
                
                num_positive = (signs > 0).sum(dim=0)
                num_negative = (signs < 0).sum(dim=0)
                num_total = len(task_vectors)
                
                agreement_ratio = torch.max(num_positive, num_negative).float() / num_total
                
                mask = agreement_ratio >= threshold
                
                total_params += mask.numel()
                kept_params += mask.sum().item()
                
                consensus_sign = torch.where(num_positive >= num_negative, 1.0, -1.0)
                
                merged_param = torch.zeros_like(param_tensors[0])
                
                if mask.any():
                    valid_values = torch.zeros_like(stacked)
                    valid_counts = torch.zeros_like(param_tensors[0])
                    
                    for i, tensor in enumerate(param_tensors):
                        agree_mask = (torch.sign(tensor) == consensus_sign) & mask
                        valid_values[i] = tensor * agree_mask.float()
                        valid_counts += agree_mask.float()
                    
                    valid_counts = torch.clamp(valid_counts, min=1.0)
                    merged_param = valid_values.sum(dim=0) / valid_counts
                
                merged_vector[param_name] = merged_param
                
                keep_ratio = mask.float().mean().item()
                logger.debug(f"{param_name}: kept {keep_ratio:.3f} of parameters")
                
            except Exception as e:
                logger.warning(f"Error processing parameter {param_name}: {e}")
                continue
        
        overall_keep_ratio = kept_params / total_params if total_params > 0 else 0
        logger.info(f"Overall kept {overall_keep_ratio:.3f} of all parameters")
        
        return merged_vector

    def create_aggressive_bias_vector(self, category: str) -> Dict[str, torch.Tensor]:
        logger.info(f"Creating AGGRESSIVE bias vector for {category}")
        
        # Task vector들 로드
        task_vectors = self.load_aggressive_task_vectors(category)
        
        if len(task_vectors) < 1:
            logger.warning(f"No aggressive task vectors found for {category}")
            return {}
        
        # Sign-merge
        bias_vector = self.sign_merge_vectors(task_vectors, self.config.sign_merge_threshold)
        
        if not bias_vector:
            logger.warning(f"Empty aggressive bias vector for {category}")
            return {}
        
        bias_vector_dir = "./aggressive_bias_vectors"
        os.makedirs(bias_vector_dir, exist_ok=True)
        bias_vector_path = f"{bias_vector_dir}/{category.lower()}_aggressive_bias_vector.pt"
        torch.save(bias_vector, bias_vector_path)
        
        logger.info(f"Aggressive bias vector saved: {bias_vector_path}")
        self.bias_vectors[category] = bias_vector
        
        return bias_vector

    def evaluate_model(self, model, category: str) -> Dict[str, float]:
        if model is None:
            return {"error": "Model is None"}

        logger.info(f"Evaluating model for {category}")

        results = {}

        # Retain set perplexity
        retain_path = self.base_data_dir / category / "retain_set.json"
        if retain_path.exists():
            try:
                retain_ppl = self.calculate_perplexity(model, str(retain_path))
                results['retain_perplexity'] = retain_ppl
                logger.info(f"{category} retain perplexity: {retain_ppl:.3f}")
            except Exception as e:
                logger.warning(f"Perplexity calculation failed for {category}: {e}")
                results['retain_perplexity'] = float('inf')

        # Test set bias score
        eval_path = self.base_data_dir / category / "eval_data.json"
        if eval_path.exists():
            try:
                bias_results = self.calculate_simple_bias_score(model, str(eval_path))
                results.update(bias_results)
                logger.info(f"{category} bias metrics calculated")
            except Exception as e:
                logger.warning(f"Bias score calculation failed for {category}: {e}")
                results.update({
                    'bias_score': 0.5,
                    'stereotype_rate': 0.5,
                    'anti_stereotype_rate': 0.5,
                    'accuracy': 0.0
                })

        return results

    def evaluate_aggressive_debiased_model(self, category: str) -> Dict[str, float]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating AGGRESSIVE debiased model for: {category}")
        logger.info(f"{'='*60}")

        bias_vector_path = f"./aggressive_bias_vectors/{category.lower()}_aggressive_bias_vector.pt"

        if not os.path.exists(bias_vector_path):
            logger.error(f"Aggressive bias vector not found: {bias_vector_path}")
            return {"error": "Aggressive bias vector not found"}

        try:
            bias_vector = torch.load(bias_vector_path, map_location='cpu')
            if not bias_vector:
                logger.error(f"Empty aggressive bias vector for {category}")
                return {"error": "Empty aggressive bias vector"}
        except Exception as e:
            logger.error(f"Failed to load aggressive bias vector: {e}")
            return {"error": f"Failed to load aggressive bias vector: {e}"}

        model_name = f"{category.lower()}_aggressive_debiased_model"
        debiased_model = self.apply_aggressive_bias_removal(bias_vector, model_name)

        if debiased_model is None:
            logger.error(f"Failed to create aggressive debiased model for {category}")
            return {"error": "Failed to create aggressive debiased model"}

        logger.info("Loading original model for comparison...")
        original_model = self.load_model_for_inference()

        results = {
            'category': category,
            'original_model': {},
            'debiased_model': {}
        }

        logger.info(f"Evaluating original model for {category}...")
        original_results = self.evaluate_model(original_model, category)
        results['original_model'] = original_results

        logger.info(f"Evaluating AGGRESSIVE debiased model for {category}...")
        debiased_results = self.evaluate_model(debiased_model, category)
        results['debiased_model'] = debiased_results

        improvements = {}

        if 'bias_score' in original_results and 'bias_score' in debiased_results:
            improvements['bias_score_reduction'] = original_results['bias_score'] - debiased_results['bias_score']

        if 'stereotype_rate' in original_results and 'stereotype_rate' in debiased_results:
            improvements['stereotype_reduction'] = original_results['stereotype_rate'] - debiased_results['stereotype_rate']

        if 'anti_stereotype_rate' in original_results and 'anti_stereotype_rate' in debiased_results:
            improvements['anti_stereotype_increase'] = debiased_results['anti_stereotype_rate'] - original_results['anti_stereotype_rate']

        if 'balance_score' in original_results and 'balance_score' in debiased_results:
            improvements['balance_improvement'] = debiased_results['balance_score'] - original_results['balance_score']

        if 'accuracy' in original_results and 'accuracy' in debiased_results:
            improvements['accuracy_change'] = debiased_results['accuracy'] - original_results['accuracy']

        if 'retain_perplexity' in original_results and 'retain_perplexity' in debiased_results:
            improvements['perplexity_change'] = debiased_results['retain_perplexity'] - original_results['retain_perplexity']

        results['improvements'] = improvements

        logger.info(f"\n🔍 {category} AGGRESSIVE Evaluation Results:")
        logger.info(f"Original Model:")
        for key, value in original_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        logger.info(f"AGGRESSIVE Debiased Model:")
        for key, value in debiased_results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        logger.info(f"AGGRESSIVE Improvements:")
        for key, value in improvements.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:+.4f}")

        del original_model, debiased_model
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def run_aggressive_evaluation_pipeline(self, max_samples_per_category: int = None):
        logger.info("Starting AGGRESSIVE Single-Axis SDU Evaluation Pipeline...")
        
        all_results = {}
        successful_categories = []
        failed_categories = []
        
        for category in self.categories:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing AGGRESSIVE category: {category}")
            logger.info(f"{'='*70}")
            
            try:
                output_dirs = self.train_category_all_runs(category, max_samples_per_category)
                
                if not output_dirs:
                    logger.error(f"No successful AGGRESSIVE training runs for {category}")
                    failed_categories.append(category)
                    continue
                
                bias_vector = self.create_aggressive_bias_vector(category)
                
                if not bias_vector:
                    logger.warning(f"Failed to create AGGRESSIVE bias vector for {category}")
                    failed_categories.append(category)
                    continue
                
                logger.info(f"✅ {category} AGGRESSIVE bias vector created successfully")
                successful_categories.append(category)
                
            except Exception as e:
                logger.error(f"❌ Error processing AGGRESSIVE {category}: {e}")
                failed_categories.append(category)
                continue
        
        logger.info(f"\n📋 AGGRESSIVE Training Summary:")
        logger.info(f"✅ Successful: {successful_categories}")
        logger.info(f"❌ Failed: {failed_categories}")
        
        if not successful_categories:
            logger.error("No successful categories for AGGRESSIVE evaluation")
            return {
                'successful_categories': successful_categories,
                'failed_categories': failed_categories,
                'results': {},
                'status': 'failed'
            }
        
        logger.info(f"\n{'='*70}")
        logger.info("Starting AGGRESSIVE Bias Removal Evaluation")
        logger.info(f"{'='*70}")
        
        for category in successful_categories:
            try:
                logger.info(f"\n🔬 Evaluating AGGRESSIVE {category} bias removal...")
                
                category_results = self.evaluate_aggressive_debiased_model(category)
                all_results[category] = category_results
                
                if 'error' not in category_results:
                    logger.info(f"✅ {category} AGGRESSIVE evaluation completed successfully")
                else:
                    logger.warning(f"⚠️ {category} AGGRESSIVE evaluation had errors: {category_results['error']}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating AGGRESSIVE {category}: {e}")
                all_results[category] = {"error": str(e)}
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info("Final AGGRESSIVE Results Summary")
        logger.info(f"{'='*70}")
        
        summary = {
            'total_categories': len(self.categories),
            'successful_training': len(successful_categories),
            'failed_training': len(failed_categories),
            'evaluation_results': {}
        }
        
        for category, results in all_results.items():
            if 'error' not in results and 'improvements' in results:
                improvements = results['improvements']
                summary['evaluation_results'][category] = {
                    'bias_score_reduction': improvements.get('bias_score_reduction', 0),
                    'stereotype_reduction': improvements.get('stereotype_reduction', 0),
                    'anti_stereotype_increase': improvements.get('anti_stereotype_increase', 0),
                    'balance_improvement': improvements.get('balance_improvement', 0),
                    'accuracy_change': improvements.get('accuracy_change', 0),
                    'perplexity_change': improvements.get('perplexity_change', 0),
                    'original_bias_score': results['original_model'].get('bias_score', 0.5),
                    'debiased_bias_score': results['debiased_model'].get('bias_score', 0.5),
                }
                
                logger.info(f"\n📊 {category} AGGRESSIVE Summary:")
                logger.info(f"   Bias Score Reduction: {improvements.get('bias_score_reduction', 0):+.4f}")
                logger.info(f"   Stereotype Reduction: {improvements.get('stereotype_reduction', 0):+.4f}")
                logger.info(f"   Anti-stereotype Increase: {improvements.get('anti_stereotype_increase', 0):+.4f}")
                logger.info(f"   Balance Improvement: {improvements.get('balance_improvement', 0):+.4f}")
                logger.info(f"   Accuracy Change: {improvements.get('accuracy_change', 0):+.4f}")
                logger.info(f"   Perplexity Change: {improvements.get('perplexity_change', 0):+.4f}")
            else:
                logger.warning(f"❌ {category}: {results.get('error', 'Unknown error')}")
        
        final_results = {
            'successful_categories': successful_categories,
            'failed_categories': failed_categories,
            'detailed_results': all_results,
            'summary': summary,
            'status': 'success' if successful_categories else 'failed'
        }
        
        results_path = "./aggressive_evaluation_results.json"
        try:
            with open(results_path, 'w') as f:
                json_results = {}
                for key, value in final_results.items():
                    if key == 'detailed_results':
                        json_results[key] = {}
                        for cat, res in value.items():
                            json_results[key][cat] = self._convert_to_json_serializable(res)
                    else:
                        json_results[key] = self._convert_to_json_serializable(value)
                
                json.dump(json_results, f, indent=2)
            logger.info(f"📄 AGGRESSIVE Results saved: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save AGGRESSIVE results: {e}")
        
        return final_results
    
    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj == float('inf') or obj == float('-inf'):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

def main():
    config = AggressiveSDUConfig(
        batch_size=4,           
        learning_rates=[1e-3, 5e-3],  
        seeds=[42, 123],        
        max_length=128,         
        num_train_epochs=10,    
        lora_r=64,              
        lora_alpha=128,         
        projection_scaling=5.0  
    )
    
    trainer = AggressiveBBQSDUTrainer(config)
    
    logger.info("🚀 Starting AGGRESSIVE BBQ-SDU Training!")
    logger.info(f"Config: LR={config.learning_rates}, Epochs={config.num_train_epochs}")
    logger.info(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info(f"Projection scaling: {config.projection_scaling}x")
    
    result = trainer.run_aggressive_evaluation_pipeline(max_samples_per_category=50)
    
    print(f"\n🎯 AGGRESSIVE Evaluation Result: {result['status']}")
    print(f"✅ Successful Categories: {result['successful_categories']}")
    print(f"❌ Failed Categories: {result['failed_categories']}")
    
    if result['status'] == 'success' and 'summary' in result:
        print(f"\n📊 AGGRESSIVE Performance Summary:")
        for category, metrics in result['summary']['evaluation_results'].items():
            print(f"   {category}:")
            print(f"     Bias Score Reduction: {metrics['bias_score_reduction']:+.4f}")
    
    return result

if __name__ == "__main__":
    main()