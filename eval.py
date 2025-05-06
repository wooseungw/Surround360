import argparse
import os
import json
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BLIP-2 with YAML config"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to eval YAML file (e.g. config/eval.yaml)"
    )
    return parser.parse_args()

class EvalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        processor: Blip2Processor,
        max_length: int,
        image_size: list,
        do_crop: bool=False,
        overlap_ratio: float=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_length = max_length
        self.do_crop = do_crop
        self.overlap_ratio = overlap_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["url"]
        query   = row["query"]
        ann     = row["annotation"]

        image = Image.open(img_path).convert("RGB")
        # (필요 시 do_crop 로직 추가)

        inputs = self.processor(
            text=query,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        pixel_values  = inputs.pixel_values.squeeze(0)   # [3, H, W]
        input_ids      = inputs.input_ids.squeeze(0)     # [L]
        attention_mask= inputs.attention_mask.squeeze(0) # [L]

        return {
            "pixel_values":  pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "url":            img_path,
            "query":          query,
            "annotation":     ann
        }

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 디바이스 설정
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 프로세서 & 모델 로드
    model_name = cfg["model"]["name_or_path"]
    processor  = Blip2Processor.from_pretrained(model_name)
    model      = Blip2ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 데이터셋 & 로더 설정
    data_dir   = cfg["data"]["dir"]
    csv_path   = os.path.join(data_dir, cfg["data"]["test_file"])
    dataset    = EvalDataset(
        csv_path=csv_path,
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

    # 생성 설정
    gen_args = {
        "max_length": cfg["generate"]["max_length"],
        "num_beams":  cfg["generate"]["num_beams"],
    }

    # 평가 지표 준비
    scorers = [
        (Bleu(4),         ["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),        "METEOR"),
        (Rouge(),         "ROUGE_L"),
        (Cider(),         "CIDEr"),
        (Spice(),         "SPICE")
    ]

    references = []
    hypotheses = []
    details    = []

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

        # 배치별 결과 축적
        for url, query, ref, pred in zip(
            batch["url"], batch["query"], batch["annotation"], preds
        ):
            references.append([ref])     # scorer 에 맞춰 list of list
            hypotheses.append(pred.strip())
            details.append({
                "url":        url,
                "query":      query,
                "reference":  ref,
                "hypothesis": pred.strip()
            })

    # 전체 스코어 계산
    overall = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(name, list):
            # BLEU_1~4
            for n, s in zip(name, score):
                overall[n] = s
        else:
            overall[name] = score

    # JSON 저장
    out = {
        "overall": overall,
        "details": details
    }
    os.makedirs(os.path.dirname(cfg["output"]["result_file"]), exist_ok=True)
    with open(cfg["output"]["result_file"], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("▶ Evaluation finished. Results saved to", cfg["output"]["result_file"])

if __name__ == "__main__":
    main()
