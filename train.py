import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Trainer
import pandas as pd
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None

import wandb
from pathlib import Path

import yaml
import argparse
from typing import Dict, List, Optional, Union, Any
import evaluate
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import clip
import torch.nn.functional as F

PAD_TOKEN_ID = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    return config

# 데이터셋 클래스 정의
class QuIC360Dataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 processor: Blip2Processor,
                 image_size: tuple = (224,224),
                 max_length: Optional[int] = None,
                 split: str = "train",
                 do_crop: bool = False,
                 fov: Optional[float] = None,
                 overlap_ratio: Optional[float] = None,
                 transform: bool = False):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length
        self.split = split
        self.do_crop = do_crop
        if self.do_crop:
            self.fov = fov
            self.overlap_ratio = overlap_ratio
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        # 이미지 경로와 질문, 정답을 가져옵니다.
        image_path = self.df.iloc[idx]["url"]
        question = self.df.iloc[idx]["query"]
        answer = self.df.iloc[idx]["annotation"]
        
        # 이미지를 로드합니다.
        image = Image.open(image_path).convert("RGB")
        
        # 질문과 정답을 전처리합니다.
        inputs = self.processor(
            text=question,
            images=image,
            image_size=self.image_size,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # 정답을 전처리합니다.
        answers = self.processor(
            text=answer,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Hugging Face Trainer가 기대하는 평평한 구조로 반환
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (C, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),        # (L1)
            "attention_mask": inputs["attention_mask"].squeeze(0),  # (L1)
            "labels": answers["input_ids"].squeeze(0),          # (L2)
            "image_path": image_path,
            "question": question,
            "answer": answer
        }

def data_collator(features):
    """Simple data collator for BLIP2"""
    # 입력 검증
    if not features:
        raise ValueError("Features list is empty!")
    
    # 첫 번째 feature 확인
    first = features[0]
    if not isinstance(first, dict):
        raise ValueError(f"Feature is not a dict, got {type(first)}")
    
    batch = {}
    
    # 텐서 필드들은 stack
    if "pixel_values" in first:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    if "input_ids" in first:
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    if "attention_mask" in first:
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    if "labels" in first:
        batch["labels"] = torch.stack([f["labels"] for f in features])
    
    # 문자열 필드들은 리스트로
    if "image_path" in first:
        batch["image_path"] = [f["image_path"] for f in features]
    if "question" in first:
        batch["question"] = [f["question"] for f in features]
    if "answer" in first:
        batch["answer"] = [f["answer"] for f in features]
    
    return batch

class CLIPScorer:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
    def compute_clip_score(self, images, texts):
        """Calculate CLIP-S score"""
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Cosine similarity
        similarity = (image_features @ text_features.T).squeeze()
        return similarity.mean().item()
    
    def compute_refclip_score(self, images, candidates, references):
        """Calculate RefCLIP-S score"""
        image_features = self.model.encode_image(images)
        candidate_features = self.model.encode_text(candidates)
        reference_features = self.model.encode_text(references)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        candidate_features = candidate_features / candidate_features.norm(dim=1, keepdim=True)
        reference_features = reference_features / reference_features.norm(dim=1, keepdim=True)
        
        # RefCLIP-S score
        candidate_similarity = (image_features @ candidate_features.T).squeeze()
        reference_similarity = (candidate_features @ reference_features.T).squeeze()
        
        refclip_score = (candidate_similarity + reference_similarity) / 2
        return refclip_score.mean().item()

def compute_metrics(eval_pred, processor, clip_scorer):
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Initialize scorers
    bleu_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    spice_scorer = Spice()
    
    # Format for pycocoevalcap
    gts = {i: [label] for i, label in enumerate(decoded_labels)}
    res = {i: [pred] for i, pred in enumerate(decoded_preds)}
    
    # Calculate scores
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    cider_score, _ = cider_scorer.compute_score(gts, res)
    spice_score, _ = spice_scorer.compute_score(gts, res)
    
    # CLIP scores - would need images which aren't available in this context
    # This is a placeholder - in actual implementation, you'd need to pass images
    clip_s = 0.0
    refclip_s = 0.0
    
    return {
        "bleu4": bleu_score[3],  # BLEU-4
        "meteor": meteor_score,
        "rouge_l": rouge_score,
        "cider": cider_score,
        "spice": spice_score,
        "clip_s": clip_s,
        "refclip_s": refclip_s
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # wandb 설정
    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'])
    
    # BLIP-2 모델 및 프로세서 로드
    model_name = config['model']['name']
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    
    # Freeze vision encoder parameters
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("Vision model parameters have been frozen.")
    # Freeze language model parameters
    for param in model.language_model.parameters():
        param.requires_grad = False
    print("Language model parameters have been frozen.")
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # CLIP scorer for evaluation
    clip_scorer = CLIPScorer(device)
    
    # 데이터셋 경로 설정
    data_dir = Path(config['data']['dir'])
    
    # 데이터셋 및 데이터로더 초기화
    image_size = tuple(config['data']['image_size'])
    max_length = config['data']['max_length']
    print("train_file:", data_dir/config['data']['train_file'])
    print("valid_file:", data_dir/config['data']['valid_file'])
    train_dataset = QuIC360Dataset(
        data_dir/config['data']['train_file'], 
        processor, 
        max_length=max_length, 
        split="train",
        image_size=image_size,
        do_crop=config['data']['do_crop'],
        overlap_ratio=config['data']['overlap_ratio']
    )
    print(f"Dataset length: {len(train_dataset)}")
    eval_dataset = QuIC360Dataset(
        data_dir/config['data']['valid_file'], 
        processor, 
        max_length=max_length, 
        split="valid",
        image_size=image_size,
        do_crop=config['data']['do_crop'],
        overlap_ratio=config['data']['overlap_ratio']
    )
    print(f"Dataset length: {len(eval_dataset)}")

    # 학습 인자 설정 - YAML 설정을 더 정확히 반영
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        run_name=config['training']['run_name'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size']['train'],
        per_device_eval_batch_size=config['training']['batch_size']['eval'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 4),
        gradient_checkpointing=config['training'].get('gradient_checkpointing', True),
        learning_rate=float(config['training'].get('learning_rate', 2e-5)),
        warmup_ratio=config['training'].get('warmup_ratio', 0.1),
        weight_decay=config['training'].get('weight_decay', 0.01),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training'].get('eval_steps', 500),
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training'].get('save_steps', 500),
        save_total_limit=config['training'].get('save_total_limit', 3),
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training'].get('metric_for_best_model', 'eval_loss'),
        greater_is_better=config['training'].get('greater_is_better', False),
        fp16=True,  # DeepSpeed config에서 관리
        deepspeed=config['deepspeed']['config'] if config['deepspeed']['enabled'] else None,
        report_to=config['training']['report_to'],
    )
    
    # Compute metrics function with processor and clip_scorer
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, processor, clip_scorer)
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )
  
    trainer.train()
    
    # 모델 저장
    save_dir = Path(config['model']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()