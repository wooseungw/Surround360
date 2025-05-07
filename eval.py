import argparse
import os
import json
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
from src.models.surroundblip import SurroundBlip
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from py360convert import e2p
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BLIP-2 모델을 YAML 설정으로 평가"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="평가용 YAML 파일 경로 (예: config/eval.yaml)"
    )
    return parser.parse_args()

class EvalDataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 processor: Blip2Processor,
                 image_size: list = [224,224],
                 max_length: Optional[int] = None,
                 do_crop: bool = False,
                 fov: Optional[float] = None,
                 overlap_ratio: Optional[float] = None,
                 transform: bool = False):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.do_crop = do_crop
        self.overlap_ratio = overlap_ratio
        if self.do_crop:
            self.image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            self.overlap_ratio = overlap_ratio
            print(f"Do Crop, Image size: {self.image_size}")
        else:
            self.image_size = tuple(image_size)
            print(f"Do not Crop, Image size: {self.image_size}")
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["url"])
        query    = str(row["query"])
        ann      = str(row["annotation"])   # 반드시 문자열로 변환

        image = Image.open(img_path).convert("RGB")
        # 필요하다면 do_crop 로직을 여기에 추가

        inputs = self.processor(
            text=query,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
                # 질문과 정답을 전처리합니다.
        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])
        
        pixel_values   = inputs.pixel_values.squeeze(0)    # [3, H, W]
        input_ids       = inputs.input_ids.squeeze(0)      # [L]
        attention_mask  = inputs.attention_mask.squeeze(0) # [L]
        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "url":            img_path,
            "query":          query,
            "annotation":     ann
        }
        
    def crop_equirectangular_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H2, W4 = img_tensor.shape
        assert B == 1
        H, W = H2 // 2, W4 // 4

        # 1) stride 각도
        step = self.fov * (1.0 - self.overlap_ratio)

        # 2) 필요한 패치 개수
        num_patches = int(np.ceil(360.0 / step))

        # 3) 0도부터 시작해 step 간격으로 중심 각 생성
        yaw_centers = (np.arange(num_patches) * step) % 360.0

        # 4) e2p u_deg 인자용으로 -180~180 범위로 매핑
        yaw_centers = np.where(yaw_centers > 180.0, yaw_centers - 360.0, yaw_centers)

        # 5) numpy array 변환
        img_np = img_tensor[0].permute(1, 2, 0).numpy()

        patches = []
        for u_deg in yaw_centers:
            pers = e2p(
                img_np,
                fov_deg=self.fov,
                u_deg=float(u_deg),
                v_deg=0.0,
                out_hw=(H, W),
                in_rot_deg=0.0,
                mode="bilinear",
            )  # (H, W, C)
            t = torch.from_numpy(pers).permute(2, 0, 1)  # (C, H, W)
            patches.append(t)

        # (N, C, H, W) → (1, N, C, H, W)
        return torch.stack(patches, dim=0).unsqueeze(0)

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 디바이스 설정
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # Processor & Model 로드
    name = cfg["model"]["name"]
    print("Model name:", name)
    pretrain_name = cfg['model']['pretrain_path']
    processor  = Blip2Processor.from_pretrained(pretrain_name)
    if name == "surround":
        print("Loading SurroundBlip model")
        # load Hugging Face BLIP-2 config and override Q-Former settings if provided
        hf_config = Blip2Config.from_pretrained(pretrain_name)
        # override top-level num_query_tokens if present
        if 'num_query_tokens' in cfg['model']:
            hf_config.num_query_tokens = cfg['model']['num_query_tokens']
        # override nested qformer_config fields if present
        if 'qformer' in cfg['model']:
            for key, value in cfg['model']['qformer'].items():
                if hasattr(hf_config.qformer_config, key):
                    setattr(hf_config.qformer_config, key, value)
        model = SurroundBlip.from_pretrained(
            pretrain_name,
            config=hf_config,
            ignore_mismatched_sizes=True
        )
    else:
        print("Loading BLIP-2 model")
        model = Blip2ForConditionalGeneration.from_pretrained(pretrain_name)
    model.to(device)
    model.eval()

    # 데이터셋 & DataLoader 설정
    data_dir = cfg["data"]["dir"]
    csv_path = os.path.join(data_dir, cfg["data"]["test_file"])
    dataset = EvalDataset(
        csv_file=csv_path,
        processor=processor,
        max_length=cfg["data"]["max_length"],
        image_size=cfg["data"]["image_size"],
        do_crop=cfg["data"].get("do_crop", False),
        overlap_ratio=cfg["data"].get("overlap_ratio", None)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"]["num_workers"],
        shuffle=False
    )

    # 생성 파라미터
    gen_args = {
        "max_new_tokens": cfg["generate"]["max_length"],
        "num_beams":  cfg["generate"]["num_beams"],
    }

    # 평가 지표 초기화
    scorers = [
        (Bleu(4),          ["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),         "METEOR"),
        (Rouge(),          "ROUGE_L"),
        (Cider(),          "CIDEr"),
        (Spice(),          "SPICE")
    ]

    references = []  # list of list of reference captions
    hypotheses = []  # list of predicted captions
    details    = []  # 샘플별 상세 정보

    # 평가 루프
    for batch in tqdm(dataloader, desc="Evaluating"):
        pv   = batch["pixel_values"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values=pv,
                input_ids=ids,
                attention_mask=mask,
                **gen_args
            )
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

        # 결과 축적
        for url, query, ref, pred in zip(
            batch["url"], batch["query"], batch["annotation"], preds
        ):
            # ref는 이미 str() 처리되어 있으므로 안전하게 사용 가능
            references.append([ref])
            hypotheses.append(pred.strip())
            details.append({
                "url":        url,
                "query":      query,
                "reference":  ref,
                "hypothesis": pred.strip()
            })

    # --- dict 포맷으로 변환 후 스코어 계산 ---
    gts = {i: references[i] for i in range(len(references))}
    res = {i: [hypotheses[i]]    for i in range(len(hypotheses))}

    overall = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(name, list):
            for n, s in zip(name, score):
                overall[n] = s
        else:
            overall[name] = score

    # 결과 JSON으로 저장
    out = {
        "overall": overall,
        "details": details
    }
    output_path = cfg["output"]["result_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"▶ 평가 완료. 결과가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()
