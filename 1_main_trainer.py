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
import bitsandbytes as bnb
import logging
import gc
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SDUConfig:
    """SDU 학습 설정 (수정됨)"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_length: int = 256
    batch_size: int = 32  # 더 작게 설정
    num_train_epochs: int = 2
    learning_rates: List[float] = None
    seeds: List[int] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = None
    sign_merge_threshold: float = 0.75
    multi_axis_threshold: float = 0.67
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [5e-5, 5e-4]
        if self.seeds is None:
            self.seeds = [42]
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

class BBQSDUTrainer:
    """BBQ-SDU 편향 제거 학습기 (수정됨)"""
    
    def __init__(self, config: SDUConfig, base_data_dir: str = "./processed_bbq"):
        self.config = config
        self.base_data_dir = Path(base_data_dir)
        self.categories = ['Race_ethnicity', 'SES', 'Gender_identity']
        
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 토크나이저 로드 (수정됨)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 결과 저장
        self.task_vectors = {}
        self.bias_vectors = {}
        
        logger.info(f"Initialized on device: {self.device}")
        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
    
    def load_model_for_training(self):
        """학습용 모델 로딩"""
        logger.info(f"Loading model for training: {self.config.model_name}")
        
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
        """추론용 모델 로딩"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=False,
            trust_remote_code=True
        )
        return model
    
    def create_lora_config(self) -> LoraConfig:
        """LoRA 설정 생성 (gradient 문제 수정)"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # *** 반드시 False ***
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.1,
            target_modules=self.config.lora_target_modules,
            bias="none"
        )
    
    def prepare_dataset(self, data_path: str, max_samples: int = None) -> Dataset:
        """데이터셋 준비 (완전 수정됨)"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples is not None:
            data = data[:max_samples]
            logger.info(f"Limited dataset to {len(data)} samples")
        
        # 텍스트만 추출
        texts = [item['text'] for item in data]
        
        def tokenize_function(examples):
            """수정된 토큰화 함수"""
            # 배치 토큰화 (패딩과 트런케이션 활성화)
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',  # 고정 길이로 패딩
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # 라벨을 input_ids의 복사본으로 설정
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # 데이터셋 생성
        dataset = Dataset.from_dict({"text": texts})
        
        # 토큰화 적용
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        
        # 첫 번째 샘플 확인
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            logger.info(f"Sample input_ids shape: {len(sample['input_ids'])}")
            logger.info(f"Sample labels shape: {len(sample['labels'])}")
            logger.info(f"Sample keys: {list(sample.keys())}")
        
        return tokenized_dataset
    
    def extract_lora_delta_weights(self, peft_model) -> Dict[str, torch.Tensor]:
        """LoRA 파라미터에서 ΔW 추출 (디버깅 강화)"""
        delta_weights = {}
        
        logger.info("Extracting LoRA delta weights...")
        
        # 전체 state dict 확인
        peft_state_dict = peft_model.state_dict()
        logger.info(f"Total parameters in state_dict: {len(peft_state_dict)}")
        
        # LoRA 관련 키들 찾기
        lora_keys = [k for k in peft_state_dict.keys() if 'lora' in k.lower()]
        logger.info(f"LoRA-related keys found: {len(lora_keys)}")
        
        # 처음 몇 개 키 출력
        all_keys = list(peft_state_dict.keys())
        logger.info(f"First 10 keys: {all_keys[:10]}")
        logger.info(f"LoRA keys: {lora_keys[:10] if lora_keys else 'None found'}")
        
        if not lora_keys:
            logger.error("No LoRA keys found in state_dict!")
            # 다른 방법으로 LoRA 파라미터 찾기 시도
            return self._extract_lora_alternative(peft_model)
        
        # LoRA A와 B 매트릭스 쌍을 찾기
        lora_pairs = {}
        for name, param in peft_state_dict.items():
            if 'lora_A' in name:
                base_name = name.replace('.lora_A.default.weight', '').replace('.lora_A.weight', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['A'] = param
                logger.debug(f"Found LoRA A: {name} -> {base_name}")
            elif 'lora_B' in name:
                base_name = name.replace('.lora_B.default.weight', '').replace('.lora_B.weight', '')
                if base_name not in lora_pairs:
                    lora_pairs[base_name] = {}
                lora_pairs[base_name]['B'] = param
                logger.debug(f"Found LoRA B: {name} -> {base_name}")
        
        logger.info(f"Found {len(lora_pairs)} LoRA pairs")
        
        # ΔW = B @ A * scaling 계산
        for base_name, matrices in lora_pairs.items():
            if 'A' in matrices and 'B' in matrices:
                lora_A = matrices['A']  # [r, in_features]
                lora_B = matrices['B']  # [out_features, r]
                
                logger.debug(f"Processing {base_name}: A{lora_A.shape} @ B{lora_B.shape}")
                
                # LoRA config에서 scaling factor 계산
                scaling = self.config.lora_alpha / self.config.lora_r
                
                # ΔW 계산
                delta_weight = (lora_B @ lora_A) * scaling
                
                # 파라미터 이름 정리
                clean_name = base_name.replace('base_model.model.', '')
                if not clean_name.endswith('.weight'):
                    clean_name += '.weight'
                
                delta_weights[clean_name] = delta_weight.cpu().float()
                logger.debug(f"Extracted {clean_name}: {delta_weight.shape}")
            else:
                logger.warning(f"Incomplete LoRA pair for {base_name}: {list(matrices.keys())}")
        
        logger.info(f"Extracted {len(delta_weights)} LoRA delta weights")
        return delta_weights
    
    def create_multi_axis_bias_vector(self) -> Dict[str, torch.Tensor]:
        """다중 축 bias vector 생성"""
        logger.info("Creating multi-axis bias vector...")
        
        # 모든 카테고리의 bias vector 로드
        category_vectors = []
        valid_categories = []
        
        for category in self.categories:
            bias_vector_path = f"./bias_vectors/{category.lower()}_bias_vector.pt"
            if os.path.exists(bias_vector_path):
                try:
                    bias_vector = torch.load(bias_vector_path, map_location='cpu')
                    if bias_vector:
                        category_vectors.append(bias_vector)
                        valid_categories.append(category)
                        logger.info(f"Loaded bias vector for {category}")
                    else:
                        logger.warning(f"Empty bias vector for {category}")
                except Exception as e:
                    logger.warning(f"Failed to load bias vector for {category}: {e}")
            else:
                logger.warning(f"Bias vector not found: {bias_vector_path}")
        
        if len(category_vectors) < 2:
            logger.error(f"Need at least 2 category bias vectors, got {len(category_vectors)}")
            return {}
        
        logger.info(f"Merging {len(category_vectors)} category vectors: {valid_categories}")
        
        # 2/3 기준으로 sign-merge
        multi_bias_vector = self.sign_merge_vectors(
            category_vectors, 
            self.config.multi_axis_threshold
        )
        
        if not multi_bias_vector:
            logger.error("Failed to create multi-axis bias vector")
            return {}
        
        # 저장
        multi_vector_path = "./bias_vectors/multi_bias_vector.pt"
        torch.save(multi_bias_vector, multi_vector_path)
        
        logger.info(f"Multi-axis bias vector saved: {multi_vector_path}")
        return multi_bias_vector
    
    def apply_bias_removal(self, bias_vector: Dict[str, torch.Tensor], 
                          output_model_name: str = "debiased_model") -> object:
        """편향 제거 적용"""
        logger.info(f"Applying bias removal: {output_model_name}")
        
        if not bias_vector:
            logger.error("Empty bias vector, cannot apply bias removal")
            return None
        
        # 원본 모델 로드
        model = self.load_model_for_inference()
        
        # Projection 적용
        with torch.no_grad():
            applied_count = 0
            total_count = len(bias_vector)
            
            for param_name, bias_delta in bias_vector.items():
                # 모델에서 해당 파라미터 찾기
                param_dict = dict(model.named_parameters())
                
                if param_name in param_dict:
                    original_param = param_dict[param_name]
                    
                    # Device와 dtype 맞추기
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        # 분산된 모델의 경우 해당 파라미터가 있는 device 찾기
                        param_device = original_param.device
                    else:
                        param_device = next(model.parameters()).device
                    
                    bias_delta_gpu = bias_delta.to(param_device, dtype=original_param.dtype)
                    
                    # Shape 일치 확인
                    if original_param.shape == bias_delta_gpu.shape:
                        # 벡터화하여 내적 계산
                        orig_flat = original_param.view(-1)
                        bias_flat = bias_delta_gpu.view(-1)
                        
                        # 내적 및 projection 계산
                        inner_product = torch.dot(orig_flat, bias_flat)
                        bias_norm_sq = torch.dot(bias_flat, bias_flat)
                        
                        if bias_norm_sq > 1e-8:
                            # Projection: proj_v(u) = <u,v>/<v,v> * v
                            projection_coef = inner_product / bias_norm_sq
                            projection = projection_coef * bias_delta_gpu
                            
                            # 편향 제거: w = w - proj_bias(w)
                            original_param.data -= projection
                            applied_count += 1
                            
                            logger.debug(f"Applied projection to {param_name}: coef={projection_coef:.6f}")
                        else:
                            logger.warning(f"Bias norm too small for {param_name}, skipping")
                    else:
                        logger.warning(f"Shape mismatch for {param_name}: {original_param.shape} vs {bias_delta_gpu.shape}")
                else:
                    logger.warning(f"Parameter {param_name} not found in model")
        
        logger.info(f"Applied bias removal to {applied_count}/{total_count} parameters")
        
        if applied_count == 0:
            logger.error("No parameters were modified during bias removal!")
            return None
        
        # 모델 저장
        output_path = f"./debiased_models/{output_model_name}"
        os.makedirs(output_path, exist_ok=True)
        
        try:
            model.save_pretrained(output_path, safe_serialization=True)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Debiased model saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
        
        return model
    
    def calculate_perplexity(self, model, data_path: str, max_samples: int = 50) -> float:
        """Perplexity 계산"""
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
                    
                    # GPU 이동
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
    
    def calculate_bias_score(self, model, data_path: str, max_samples: int = 100) -> float:
        """BBQ bias score 계산"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            return 0.0
        
        model.eval()
        correct_fair = 0
        total_valid = 0
        
        with torch.no_grad():
            for i, item in enumerate(data[:max_samples]):
                try:
                    # 필수 필드 확인
                    required_fields = ['context', 'question', 'ans0', 'ans1', 'ans2', 'answer_label', 'target_label']
                    if not all(field in item for field in required_fields):
                        continue
                    
                    context = item['context']
                    question = item['question']
                    answers = [item['ans0'], item['ans1'], item['ans2']]
                    
                    # 프롬프트 구성
                    base_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
                    
                    # 각 답변의 확률 계산
                    answer_scores = []
                    
                    # GPU device 찾기
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        device = list(model.hf_device_map.values())[0]
                    else:
                        device = next(model.parameters()).device
                    
                    for answer in answers:
                        full_prompt = base_prompt + f" {answer}"
                        
                        inputs = self.tokenizer(
                            full_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.config.max_length
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = model(**inputs)
                        
                        # 답변 부분의 평균 로그 확률 계산
                        base_inputs = self.tokenizer(base_prompt, return_tensors="pt")
                        base_len = base_inputs['input_ids'].shape[1]
                        
                        if inputs['input_ids'].shape[1] > base_len:
                            answer_logits = outputs.logits[0, base_len-1:-1, :]
                            answer_tokens = inputs['input_ids'][0, base_len:]
                            
                            log_probs = F.log_softmax(answer_logits, dim=-1)
                            token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
                            avg_log_prob = token_log_probs.mean().item()
                            answer_scores.append(avg_log_prob)
                        else:
                            answer_scores.append(-float('inf'))
                    
                    # 예측된 답변
                    if answer_scores and not all(score == -float('inf') for score in answer_scores):
                        predicted_idx = np.argmax(answer_scores)
                        answer_label = item['answer_label']
                        target_label = item['target_label']
                        
                        # 정답이면서 편향되지 않은 경우
                        if predicted_idx == answer_label and answer_label != target_label:
                            correct_fair += 1
                        
                        total_valid += 1
                        
                except Exception as e:
                    logger.warning(f"Bias score calculation error for sample {i}: {e}")
                    continue
        
        if total_valid == 0:
            logger.warning("No valid samples for bias score calculation")
            return 0.0
        
        bias_score = correct_fair / total_valid
        return bias_score
    
    def evaluate_model(self, model, category: str) -> Dict[str, float]:
        """모델 평가"""
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
                bias_score = self.calculate_bias_score(model, str(eval_path))
                results['bias_score'] = bias_score
                logger.info(f"{category} bias score: {bias_score:.3f}")
            except Exception as e:
                logger.warning(f"Bias score calculation failed for {category}: {e}")
                results['bias_score'] = 0.0
        
        return results
    
    def _extract_lora_alternative(self, peft_model) -> Dict[str, torch.Tensor]:
        """대안적인 LoRA 가중치 추출 방법"""
        logger.info("Trying alternative LoRA extraction...")
        
        delta_weights = {}
        
        try:
            # PEFT 모델에서 직접 trainable 파라미터 접근
            trainable_params = {}
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    trainable_params[name] = param.detach().cpu()
                    logger.debug(f"Found trainable LoRA param: {name} - {param.shape}")
            
            logger.info(f"Found {len(trainable_params)} trainable LoRA parameters")
            
            # A, B 매트릭스 쌍 찾기
            lora_pairs = {}
            for name, param in trainable_params.items():
                if '.lora_A.' in name:
                    base_name = name.split('.lora_A.')[0]
                    if base_name not in lora_pairs:
                        lora_pairs[base_name] = {}
                    lora_pairs[base_name]['A'] = param
                elif '.lora_B.' in name:
                    base_name = name.split('.lora_B.')[0]
                    if base_name not in lora_pairs:
                        lora_pairs[base_name] = {}
                    lora_pairs[base_name]['B'] = param
            
            logger.info(f"Alternative method found {len(lora_pairs)} LoRA pairs")
            
            # ΔW 계산
            scaling = self.config.lora_alpha / self.config.lora_r
            for base_name, matrices in lora_pairs.items():
                if 'A' in matrices and 'B' in matrices:
                    lora_A = matrices['A']
                    lora_B = matrices['B']
                    
                    delta_weight = (lora_B @ lora_A) * scaling
                    
                    # 이름 정리
                    clean_name = base_name.replace('base_model.model.', '') + '.weight'
                    delta_weights[clean_name] = delta_weight.float()
                    
                    logger.debug(f"Alternative extracted {clean_name}: {delta_weight.shape}")
            
        except Exception as e:
            logger.error(f"Alternative extraction failed: {e}")
        
        return delta_weights
    
    def train_single_run(self, category: str, lr: float, seed: int, max_samples: int = None) -> str:
        """단일 학습 실행 (gradient 문제 수정됨)"""
        run_name = f"{category.lower()}_lr{lr}_seed{seed}"
        output_dir = f"./checkpoints/{run_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting training: {run_name}")
        
        # 시드 설정
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            # 데이터 준비
            forget_data_path = self.base_data_dir / category / "forget_set.json"
            if not forget_data_path.exists():
                logger.error(f"Data file not found: {forget_data_path}")
                return ""
                
            dataset = self.prepare_dataset(str(forget_data_path), max_samples)
            
            # 데이터셋 검증
            if len(dataset) == 0:
                logger.error("Empty dataset")
                return ""
            
            # 모델 로드
            model = self.load_model_for_training()
            
            # LoRA 적용
            lora_config = self.create_lora_config()
            model = get_peft_model(model, lora_config)
            
            # *** 중요: 모델을 명시적으로 훈련 모드로 설정 ***
            model.train()
            
            # *** LoRA 파라미터 gradient 활성화 확인 ***
            trainable_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    logger.debug(f"Trainable: {name} - {param.shape}")
                else:
                    logger.debug(f"Frozen: {name} - {param.shape}")
            
            logger.info(f"Trainable params: {trainable_params:,} / Total: {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
            if trainable_params == 0:
                logger.error("No trainable parameters found!")
                return ""
            
            # 학습 설정 (gradient checkpointing 비활성화)
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=lr,
                fp16=True,
                save_strategy="no",
                dataloader_drop_last=True,
                gradient_checkpointing=False,  # *** LoRA와 충돌 방지 ***
                dataloader_pin_memory=False,
                seed=seed,
                remove_unused_columns=False,
                report_to=None,
                warmup_steps=0,
                weight_decay=0.01,
                gradient_accumulation_steps=1,
                # 추가된 설정들
                logging_steps=10,  # 더 자주 로깅
                eval_steps=None,
                save_steps=None,
                dataloader_num_workers=0,  # 멀티프로세싱 비활성화
            )
            
            # 수정된 데이터 콜레이터
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # 트레이너 생성
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            
            # *** 기본 옵티마이저 사용 (더 안전함) ***
            # Trainer가 자동으로 AdamW 옵티마이저를 생성하도록 함
            logger.info(f"Using default trainer optimizer with lr={lr}")
            
            # 학습 실행
            start_time = time.time()
            logger.info("Starting training process...")
            
            # *** 첫 스텝에서 gradient 확인 ***
            model.train()  # 다시 한번 확인
            
            trainer.train()
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # LoRA 델타 가중치 추출
            delta_weights = self.extract_lora_delta_weights(model)
            
            if not delta_weights:
                logger.error("No delta weights extracted!")
                return ""
            
            # 저장
            torch.save(delta_weights, f"{output_dir}/task_vector.pt")
            
            # 메타데이터 저장
            metadata = {
                'category': category,
                'learning_rate': lr,
                'seed': seed,
                'training_time': training_time,
                'dataset_size': len(dataset),
                'num_parameters': len(delta_weights)
            }
            with open(f"{output_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training completed successfully: {run_name}")
            
        except Exception as e:
            logger.error(f"Training failed for {run_name}: {str(e)}")
            return ""
        
        finally:
            # 메모리 정리
            try:
                del model, trainer, optimizer
            except:
                pass
            torch.cuda.empty_cache()
            gc.collect()
        
        return output_dir
    
    def train_category_all_runs(self, category: str, max_samples: int = None) -> List[str]:
        """카테고리별 모든 조합 학습"""
        logger.info(f"Training all runs for category: {category}")
        
        output_dirs = []
        total_runs = len(self.config.learning_rates) * len(self.config.seeds)
        current_run = 0
        
        for lr in self.config.learning_rates:
            for seed in self.config.seeds:
                current_run += 1
                logger.info(f"Run {current_run}/{total_runs}: lr={lr}, seed={seed}")
                
                output_dir = self.train_single_run(category, lr, seed, max_samples)
                if output_dir:  # 성공한 경우만 추가
                    output_dirs.append(output_dir)
                    logger.info(f"✅ Successful run: {lr}, {seed}")
                else:
                    logger.warning(f"❌ Failed run: {category}, lr={lr}, seed={seed}")
        
        logger.info(f"Completed {len(output_dirs)}/{total_runs} runs for {category}")
        return output_dirs
    
    def load_task_vectors(self, category: str) -> List[Dict[str, torch.Tensor]]:
        """카테고리의 모든 task vector 로드"""
        task_vectors = []
        
        for lr in self.config.learning_rates:
            for seed in self.config.seeds:
                run_name = f"{category.lower()}_lr{lr}_seed{seed}"
                vector_path = f"./checkpoints/{run_name}/task_vector.pt"
                
                if os.path.exists(vector_path):
                    try:
                        task_vector = torch.load(vector_path, map_location='cpu')
                        if task_vector:  # 빈 딕셔너리가 아닌 경우만
                            task_vectors.append(task_vector)
                            logger.debug(f"Loaded task vector: {run_name}")
                        else:
                            logger.warning(f"Empty task vector: {run_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {vector_path}: {e}")
                else:
                    logger.warning(f"Task vector not found: {vector_path}")
        
        logger.info(f"Loaded {len(task_vectors)} valid task vectors for {category}")
        return task_vectors
    
    def sign_merge_vectors(self, task_vectors: List[Dict[str, torch.Tensor]], 
                          threshold: float = None) -> Dict[str, torch.Tensor]:
        """Sign-merge 알고리즘"""
        if not task_vectors:
            logger.warning("No task vectors to merge")
            return {}
        
        if len(task_vectors) == 1:
            logger.info("Only one vector, returning as-is")
            return task_vectors[0]
        
        if threshold is None:
            threshold = self.config.sign_merge_threshold
            
        logger.info(f"Sign-merging {len(task_vectors)} vectors with threshold {threshold}")
        
        # 공통 파라미터 이름 추출
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
                # 모든 벡터에서 해당 파라미터 스택
                param_tensors = []
                for vector in task_vectors:
                    if param_name in vector:
                        param_tensors.append(vector[param_name])
                
                if len(param_tensors) != len(task_vectors):
                    logger.warning(f"Inconsistent parameter {param_name}, skipping")
                    continue
                
                stacked = torch.stack(param_tensors)  # [num_vectors, ...]
                
                # 부호 계산
                signs = torch.sign(stacked)
                
                # 각 위치별로 양수/음수 개수 계산
                num_positive = (signs > 0).sum(dim=0)
                num_negative = (signs < 0).sum(dim=0)
                num_total = len(task_vectors)
                
                # 더 많은 쪽의 비율 계산
                agreement_ratio = torch.max(num_positive, num_negative).float() / num_total
                
                # 임계값 이상인 위치만 선택
                mask = agreement_ratio >= threshold
                
                # 통계 업데이트
                total_params += mask.numel()
                kept_params += mask.sum().item()
                
                # 합의된 부호 결정
                consensus_sign = torch.where(num_positive >= num_negative, 1.0, -1.0)
                
                # 마스크된 위치에서 합의된 부호와 같은 값들만 평균
                merged_param = torch.zeros_like(param_tensors[0])
                
                if mask.any():
                    valid_values = torch.zeros_like(stacked)
                    valid_counts = torch.zeros_like(param_tensors[0])
                    
                    for i, tensor in enumerate(param_tensors):
                        # 부호가 일치하고 마스크가 True인 위치
                        agree_mask = (torch.sign(tensor) == consensus_sign) & mask
                        valid_values[i] = tensor * agree_mask.float()
                        valid_counts += agree_mask.float()
                    
                    # 평균 계산 (0으로 나누기 방지)
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
    
    def create_bias_vector(self, category: str) -> Dict[str, torch.Tensor]:
        """카테고리별 bias vector 생성"""
        logger.info(f"Creating bias vector for {category}")
        
        # Task vector들 로드
        task_vectors = self.load_task_vectors(category)
        
        if len(task_vectors) < 1:
            logger.warning(f"No task vectors found for {category}")
            return {}
        
        # Sign-merge (단일 벡터인 경우 그대로 사용)
        bias_vector = self.sign_merge_vectors(task_vectors, self.config.sign_merge_threshold)
        
        if not bias_vector:
            logger.warning(f"Empty bias vector for {category}")
            return {}
        
        # 저장
        bias_vector_dir = "./bias_vectors"
        os.makedirs(bias_vector_dir, exist_ok=True)
        bias_vector_path = f"{bias_vector_dir}/{category.lower()}_bias_vector.pt"
        torch.save(bias_vector, bias_vector_path)
        
        logger.info(f"Bias vector saved: {bias_vector_path}")
        self.bias_vectors[category] = bias_vector
        
        return bias_vector
    
    def run_full_pipeline(self, max_samples_per_category: int = None):
        """전체 파이프라인 실행"""
        logger.info("Starting full SDU pipeline...")
        
        successful_categories = []
        failed_categories = []
        
        # 1. 각 카테고리별 학습 및 bias vector 생성
        for category in self.categories:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing category: {category}")
            logger.info(f"{'='*50}")
            
            try:
                # 모든 조합 학습
                output_dirs = self.train_category_all_runs(category, max_samples_per_category)
                
                if not output_dirs:
                    logger.error(f"No successful training runs for {category}")
                    failed_categories.append(category)
                    continue
                
                # Bias vector 생성
                bias_vector = self.create_bias_vector(category)
                
                if bias_vector:
                    logger.info(f"✅ {category} bias vector created successfully")
                    successful_categories.append(category)
                else:
                    logger.warning(f"⚠️ Failed to create bias vector for {category}")
                    failed_categories.append(category)
                    
            except Exception as e:
                logger.error(f"❌ Error processing {category}: {e}")
                failed_categories.append(category)
                continue
        
        logger.info(f"\nProcessing summary:")
        logger.info(f"✅ Successful: {successful_categories}")
        logger.info(f"❌ Failed: {failed_categories}")
        
        if len(successful_categories) >= 1:  # 최소 1개만 있어도 진행
            logger.info(f"Found {len(successful_categories)} successful categories, proceeding...")
            
            # 2. Multi-axis bias vector 생성 (2개 이상일 때만)
            if len(successful_categories) >= 2:
                logger.info(f"\n{'='*50}")
                logger.info("Creating multi-axis bias vector")
                logger.info(f"{'='*50}")
                
                try:
                    multi_bias_vector = self.create_multi_axis_bias_vector()
                    
                    # 3. 최종 평가
                    if multi_bias_vector:
                        logger.info(f"\n{'='*50}")
                        logger.info("Creating final debiased model")
                        logger.info(f"{'='*50}")
                        
                        final_model = self.apply_bias_removal(
                            multi_bias_vector, "final_debiased_model"
                        )
                        
                        if final_model:
                            # 각 카테고리별 평가
                            final_results = {}
                            for category in successful_categories:
                                try:
                                    eval_results = self.evaluate_model(final_model, category)
                                    final_results[category] = eval_results
                                    logger.info(f"{category} final results: {eval_results}")
                                except Exception as e:
                                    logger.warning(f"Evaluation failed for {category}: {e}")
                                    final_results[category] = {"error": str(e)}
                            
                            # 결과 저장
                            results_path = "./final_evaluation_results.json"
                            with open(results_path, 'w') as f:
                                json.dump(final_results, f, indent=2)
                            
                            logger.info(f"📊 Final results saved: {results_path}")
                            
                            # 메모리 정리
                            del final_model
                            torch.cuda.empty_cache()
                        else:
                            logger.error("❌ Failed to create final debiased model")
                    else:
                        logger.error("❌ Failed to create multi-axis bias vector")
                        
                except Exception as e:
                    logger.error(f"❌ Error in multi-axis processing: {e}")
            else:
                logger.info("Only one successful category, skipping multi-axis processing")
                logger.info("Individual bias vectors created successfully!")
            
            # 성공한 카테고리들로 처리 진행
            return {
                'successful_categories': successful_categories,
                'failed_categories': failed_categories,
                'status': 'partial_success' if failed_categories else 'success'
            }
        else:
            logger.error("No successful categories for processing")
            return {
                'successful_categories': successful_categories,
                'failed_categories': failed_categories,
                'status': 'failed'
            }

def main():
    """메인 실행 함수 (디버깅용 설정)"""
    # 작은 설정으로 테스트
    config = SDUConfig(
        batch_size=8,   # 더 작게
        learning_rates=[5e-5],  # 하나만 테스트
        seeds=[42],
        max_length=64   # 더 짧게
    )
    
    trainer = BBQSDUTrainer(config)
    
    # *** 간단한 gradient 테스트 먼저 ***
    logger.info("Testing LoRA setup...")
    test_result = test_lora_gradient(trainer)
    if not test_result:
        logger.error("LoRA gradient test failed!")
        return
    
    # 소수 샘플로 테스트
    result = trainer.run_full_pipeline(max_samples_per_category=50)  # 더 작게
    
    print(f"\n🎯 Final Result: {result['status']}")
    print(f"✅ Successful: {result['successful_categories']}")
    print(f"❌ Failed: {result['failed_categories']}")
    
    return result

def test_lora_gradient(trainer: BBQSDUTrainer) -> bool:
    """LoRA gradient 설정 테스트"""
    try:
        # 작은 모델로 테스트
        model = trainer.load_model_for_training()
        lora_config = trainer.create_lora_config()
        model = get_peft_model(model, lora_config)
        
        model.train()
        
        # 파라미터 상태 확인
        trainable_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            total_count += 1
            if param.requires_grad:
                trainable_count += 1
                logger.info(f"✅ Trainable: {name}")
            else:
                logger.debug(f"❄️ Frozen: {name}")
        
        logger.info(f"Trainable: {trainable_count}/{total_count} parameters")
        
        if trainable_count == 0:
            logger.error("No trainable parameters!")
            return False
        
        # 간단한 forward pass 테스트
        dummy_input = torch.randint(0, 1000, (1, 10)).to(trainer.device)
        dummy_labels = dummy_input.clone()
        
        outputs = model(input_ids=dummy_input, labels=dummy_labels)
        loss = outputs.loss
        
        logger.info(f"Test loss: {loss.item():.4f}")
        
        # Backward pass 테스트
        loss.backward()
        
        # Gradient 확인
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        logger.info(f"Parameters with gradients: {grad_count}")
        
        # 정리
        del model
        torch.cuda.empty_cache()
        
        return grad_count > 0
        
    except Exception as e:
        logger.error(f"Gradient test failed: {e}")
        return False

if __name__ == "__main__":
    main()