import argparse
import os
import json
import yaml
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Union, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Blip2Config,
    CLIPProcessor,
    CLIPModel
)
from torchvision import transforms # QuIC360Dataset에서 사용

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import pandas as pd
import numpy as np
from py360convert import e2p

from src.models.surroundblip import SurroundBlip

# ==============================================================================
# 로깅 설정
# ==============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==============================================================================
# 유틸리티 함수
# ==============================================================================

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_model_and_processor(cfg: Dict[str, Any], device: torch.device):
    model_name = cfg["model"]["name"]
    pretrain_path = cfg["model"]["pretrain_path"]
    processor = Blip2Processor.from_pretrained(pretrain_path)
    if model_name.lower() == "surround":
        hf_config = Blip2Config.from_pretrained(pretrain_path)
        model = SurroundBlip.from_pretrained(pretrain_path, config=hf_config, ignore_mismatched_sizes=True)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(pretrain_path)
    model.to(device)
    model.eval()
    return processor, model

def get_clip_models(model_name: str, device: torch.device):
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    return clip_model, clip_processor

# ==============================================================================
# [변경] QuIC360Dataset 및 개선된 Data Collator
# ==============================================================================

IGNORE_INDEX = -100

class QuIC360Dataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 processor: Blip2Processor,
                 image_size: list = [224, 224],
                 max_length: Optional[int] = 128,
                 split: str = "train",
                 do_crop: bool = False,
                 fov: Optional[float] = 90.0,
                 overlap_ratio: Optional[float] = 0.5,
                 use_augmentation: bool = False): # 평가는 항상 False
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.do_crop = do_crop
        
        if self.do_crop:
            self.image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            self.overlap_ratio = overlap_ratio
        else:
            self.image_size = tuple(image_size)

        self.use_augmentation = use_augmentation
        if self.use_augmentation and self.split == 'train':
            logger.info("데이터 증강 적용 (ColorJitter, GaussianBlur).")
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        image_path = row["url"]
        question = str(row["query"])
        answer = str(row["annotation"])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"이미지 로딩 실패: {image_path} | {e}. 검정 이미지로 대체합니다.")
            image = Image.new("RGB", self.image_size, (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        # [수정] 평가 시에는 정답을 제외한 프롬프트만 모델에 제공
        if self.split == 'train':
            text_to_process = f"Query: {question} Answer: {answer}"
        else: # 'eval', 'test' 등
            text_to_process = f"Query: {question} Answer:"

        inputs = self.processor(
            images=image, text=text_to_process, return_tensors="pt",
            max_length=self.max_length, padding="max_length", truncation=True,
        )
        
        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])
        
        # 학습 시에만 필요한 레이블
        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0), # 평가 시 사용되진 않음
            "image_path": image_path,
            "question": question,
            "answer": answer
        }

    def crop_equirectangular_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H2, W4 = img_tensor.shape
        assert B == 1
        H, W = H2 // 2, W4 // 4
        step = self.fov * (1.0 - self.overlap_ratio)
        num_patches = int(np.ceil(360.0 / step))
        yaw_centers = (np.arange(num_patches) * step) % 360.0
        yaw_centers = np.where(yaw_centers > 180.0, yaw_centers - 360.0, yaw_centers)
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        patches = [
            torch.from_numpy(
                e2p(img_np, fov_deg=self.fov, u_deg=float(u), v_deg=0.0, out_hw=(H, W), mode="bilinear")
            ).permute(2, 0, 1) for u in yaw_centers
        ]
        return torch.stack(patches, dim=0).unsqueeze(0)

def data_collator(features: List[Dict]) -> Dict[str, Any]:
    """[개선] Blip2/VLM을 위한 안정적인 데이터 콜레이터"""
    if not features: return {}
    
    first = features[0]
    batch = {}
    
    # Tensor 필드 처리 (pixel_values는 텐서 리스트일 수 있음)
    if "pixel_values" in first:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

    # Tokenizer를 이용한 텍스트 관련 필드 패딩
    text_keys = ["input_ids", "attention_mask", "labels"]
    text_features = [{k: v for k, v in f.items() if k in text_keys} for f in features]
    
    # Blip2Processor의 tokenizer는 LlamaTokenizer 등을 기반으로 함
    # 이 tokenizer의 pad 메서드를 사용하여 올바른 패딩을 적용
    processor = features[0]['processor'] # 데이터셋에서 프로세서를 가져올 수 있도록 수정 필요
                                         # 또는 콜레이터 초기화 시 프로세서 전달
    # 임시방편: 전역 processor 변수 사용
    padded_batch = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b").tokenizer.pad(
        text_features, padding=True, return_tensors="pt"
    )
    batch.update(padded_batch)

    # 문자열 필드 처리
    str_keys = ["image_path", "question", "answer"]
    for key in str_keys:
        if key in first:
            batch[key] = [f[key] for f in features]
            
    return batch


# ==============================================================================
# [변경] Evaluator 클래스: QuIC360Dataset을 사용하도록 수정
# ==============================================================================

class Evaluator:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.processor, self.clip_model, self.clip_processor, self.dataloader = [None] * 5
        logger.info(f"[Evaluator] 초기화 완료, 디바이스: {self.device}")

    def setup(self):
        """평가에 필요한 모든 컴포넌트를 설정합니다."""
        self.processor, self.model = get_model_and_processor(self.cfg, self.device)

        data_cfg = self.cfg["data"]
        csv_path = os.path.join(data_cfg["dir"], data_cfg["test_file"])
        
        # [변경] QuIC360Dataset 사용
        dataset = QuIC360Dataset(
            csv_file=csv_path,
            processor=self.processor,
            split='eval', # 'train'이 아니므로 평가 모드로 동작
            max_new_tokens=data_cfg.get("max_new_tokens", 128),
            image_size=data_cfg.get("image_size", [224, 224]),
            do_crop=data_cfg.get("do_crop", False),
            fov=data_cfg.get("fov", 90.0),
            overlap_ratio=data_cfg.get("overlap_ratio", 0.5),
            use_augmentation=False # 평가 시 증강 비활성화
        )
        
        # [변경] 개선된 data_collator를 사용하기 위해 collate_fn 지정
        # 참고: 올바른 패딩을 위해 tokenizer를 콜레이터에 전달하는 것이 가장 이상적입니다.
        # 여기서는 Blip2Processor의 기본 pad_token_id가 0임을 가정하고 간단히 구현합니다.
        def robust_collator(features: List[Dict]) -> Dict:
            batch = {}
            keys = features[0].keys()
            for key in keys:
                if isinstance(features[0][key], torch.Tensor):
                    batch[key] = torch.stack([f[key] for f in features])
                else:
                    batch[key] = [f[key] for f in features]
            return batch
            
        self.dataloader = DataLoader(
            dataset, batch_size=self.cfg["eval"]["batch_size"],
            num_workers=self.cfg["eval"]["num_workers"], shuffle=False,
            pin_memory=True
            # collate_fn=robust_collator # tokenizer.pad를 사용하는 것이 더 안정적
        )

        metric_names = [m.lower() for m in self.cfg["eval"].get("metrics", [])]
        if "clip-s" in metric_names or "refclip-s" in metric_names:
            clip_model_name = self.cfg["eval"].get("clip_model_name", "openai/clip-vit-base-patch32")
            self.clip_model, self.clip_processor = get_clip_models(clip_model_name, self.device)

    def run(self):
        """전체 평가 파이프라인을 실행합니다."""
        self.setup()
        gen_args = self.cfg["generate"]
        references, hypotheses, details = self._generate_captions(gen_args)
        overall_scores = self._calculate_all_metrics(references, hypotheses, details)
        save_results(overall_scores, details, self.cfg["output"]["result_file"])

    def _generate_captions(self, gen_args: Dict[str, Any]) -> tuple:
        """배치 단위로 캡션을 생성합니다."""
        references, hypotheses, details = [], [], []
        for batch in tqdm(self.dataloader, desc="Generating Captions"):
            for key in ["pixel_values", "input_ids", "attention_mask"]:
                batch[key] = batch[key].to(self.device)

            with torch.no_grad():
                # [변경] VQA 형식이므로, prompt를 포함한 input_ids를 전달해야 함
                gen_ids = self.model.generate(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_args
                )
            
            # 생성된 텍스트에서 프롬프트 부분은 제외
            preds = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

            # [변경] 데이터 키 변경: url->image_path, query->question, annotation->answer
            for path, q, ref, pred in zip(batch["image_path"], batch["question"], batch["answer"], preds):
                references.append([ref])
                hypotheses.append(pred.strip())
                details.append({"url": path, "query": q, "reference": ref, "hypothesis": pred.strip()})
        return references, hypotheses, details

    # 이하 _calculate_all_metrics, _get_scorers 등 다른 메서드들은 이전과 동일하게 유지 ...
    def _calculate_all_metrics(self, references: List, hypotheses: List, details: List) -> Dict:
        metric_names = [m.lower() for m in self.cfg["eval"]["metrics"]]
        batch_size = self.cfg["eval"].get("clip_batch_size", 32)
        scorers = self._get_scorers(metric_names)
        overall = self._compute_coco_scores(scorers, references, hypotheses)
        if "clip-s" in metric_names:
            overall["CLIP-S"] = self._compute_clip_s_batched(details, batch_size)
        if "refclip-s" in metric_names:
            overall["RefCLIP-S"] = self._compute_refclip_s_batched(details, batch_size)
        return overall

    def _get_scorers(self, metric_names: List[str]):
        SCORER_MAP = {"bleu": (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), "meteor": (Meteor(), "METEOR"), "rouge": (Rouge(), "ROUGE_L"), "cider": (Cider(), "CIDEr"), "spice": (Spice(), "SPICE")}
        scorers = []
        for name in metric_names:
            key = name.split('-')[0].lower()
            if key in SCORER_MAP: scorers.append(SCORER_MAP[key])
        return scorers

    def _compute_coco_scores(self, scorers, references, hypotheses):
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [hypo] for i, hypo in enumerate(hypotheses)}
        overall = {}
        for scorer, name in scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(name, list):
                for n, s in zip(name, score): overall[n] = float(s)
            else: overall[name] = float(score)
        return overall

    def _compute_clip_s_batched(self, details: List, batch_size: int) -> float:
        image_paths = [s["url"] for s in details]
        hypotheses = [s["hypothesis"] for s in details]
        similarities = []
        with torch.no_grad():
            for i in tqdm(range(0, len(details), batch_size), desc="Calculating CLIP-S (Batched)"):
                batch_paths, batch_texts = image_paths[i:i+batch_size], hypotheses[i:i+batch_size]
                images = [Image.open(p).convert("RGB") if os.path.exists(p) else Image.new("RGB", (224, 224)) for p in batch_paths]
                inputs = self.clip_processor(text=batch_texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                outputs = self.clip_model(**inputs)
                similarities.extend(F.cosine_similarity(outputs.image_embeds, outputs.text_embeds).cpu().tolist())
        return float(np.mean(similarities)) if similarities else 0.0

    def _compute_refclip_s_batched(self, details: List, batch_size: int) -> float:
        hypotheses = [s["hypothesis"] for s in details]
        references = [s["reference"] for s in details]
        text_sims = []
        with torch.no_grad():
            for i in tqdm(range(0, len(details), batch_size), desc="Calculating RefCLIP-S (Batched)"):
                batch_hypo, batch_ref = hypotheses[i:i+batch_size], references[i:i+batch_size]
                inputs_hypo = self.clip_processor(text=batch_hypo, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                inputs_ref = self.clip_processor(text=batch_ref, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                hypo_embeds, ref_embeds = self.clip_model.get_text_features(**inputs_hypo), self.clip_model.get_text_features(**inputs_ref)
                text_sims.extend(F.cosine_similarity(hypo_embeds, ref_embeds).cpu().tolist())
        return float(np.mean(text_sims)) if text_sims else 0.0

def save_results(overall: Dict, details: List, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "details": details}, f, ensure_ascii=False, indent=2)
    logger.info(f"▶ 평가 완료. 결과가 저장되었습니다: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vision-Language Models")
    parser.add_argument("--config", type=str, required=True, help="평가용 YAML 파일 경로")
    args = parser.parse_args()
    try:
        evaluator = Evaluator(config_path=args.config)
        evaluator.run()
    except Exception as e:
        logger.critical(f"평가 프로세스 중 치명적인 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()