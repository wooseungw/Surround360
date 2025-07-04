import argparse
import os
import json
import yaml
import logging
from tqdm import tqdm

import torch
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
from src.models.surroundblip import SurroundBlip

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import pandas as pd
from py360convert import e2p
import numpy as np
from typing import Dict, List, Optional, Any

# -----------------------------------------------------------------------------------
# 1) 로깅 설정
# -----------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -----------------------------------------------------------------------------------
# 2) 구성 파일 로드 및 검증
# -----------------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """
    YAML 형식의 설정 파일을 안전하게 로드하고, 필수 키가 존재하는지 검증한다.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 필수 키 검증
    required_keys = [
        ("model", "name"),
        ("model", "pretrain_path"),
        ("data", "dir"),
        ("data", "test_file"),
        ("generate", "max_length"),
        ("generate", "num_beams"),
        ("output", "result_file"),
        ("eval", "batch_size"),
        ("eval", "num_workers")
    ]
    for section, key in required_keys:
        if section not in cfg or key not in cfg[section]:
            raise KeyError(f"config 파일에 '{section}.{key}' 설정이 누락되었습니다.")
    return cfg

# -----------------------------------------------------------------------------------
# 3) 데이터셋 정의
# -----------------------------------------------------------------------------------
class EvalDataset(Dataset):
    """
    CSV 파일에 기록된 샘플 단위로 이미지 경로, 질의(query), 정답(annotation) 정보를 읽어들여
    BLIP-2 Processor에 맞게 전처리하고, 필요 시 큐브맵 크롭을 수행한 후,
    배치 추론 단계에서 사용할 딕셔너리 형태로 반환한다.
    """
    def __init__(
        self,
        csv_file: str,
        processor: Blip2Processor,
        image_size: List[int] = [224, 224],
        max_length: Optional[int] = None,
        do_crop: bool = False,
        fov: Optional[float] = None,
        overlap_ratio: Optional[float] = None
    ):
        super().__init__()
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.do_crop = do_crop
        self.overlap_ratio = overlap_ratio

        if self.do_crop:
            # 파노라마 이미지를 2배 x 4배 해상도로 조정 (equirectangular 크롭용)
            self.image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            logger.info(f"[데이터셋] 크롭 사용, 이미지 사이즈: {self.image_size}")
        else:
            self.image_size = tuple(image_size)
            logger.info(f"[데이터셋] 크롭 미사용, 이미지 사이즈: {self.image_size}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row["url"])
        query = str(row["query"])
        ann = str(row["annotation"])  # 문자열로 변환

        # 이미지 로딩 예외 처리
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"이미지 로딩 실패: {img_path} | 오류: {e}")
            # 실패한 샘플은 검정색 배경 이미지로 대체
            image = Image.new("RGB", self.image_size, (0, 0, 0))

        # BLIP-2 Processor 전처리
        inputs = self.processor(
            text=query,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # 파노라마 크롭이 필요한 경우
        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])

        pixel_values = inputs.pixel_values.squeeze(0)     # [3, H, W] 또는 [N, C, H, W] (크롭 시)
        input_ids = inputs.input_ids.squeeze(0)           # [L]
        attention_mask = inputs.attention_mask.squeeze(0) # [L]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "url": img_path,
            "query": query,
            "annotation": ann
        }

    def crop_equirectangular_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        equirectangular 이미지(1 x C x H2 x W4)에서 지정된 FOV, overlap 비율을 기반으로
        여러 패치를 크롭하여 리턴한다. (N x C x H x W 포맷)
        """
        B, C, H2, W4 = img_tensor.shape
        assert B == 1, "batch 차원이 1이어야만 equirectangular 크롭이 가능합니다."
        H, W = H2 // 2, W4 // 4

        # (1) stride 각도 계산
        step = self.fov * (1.0 - self.overlap_ratio)
        # (2) 필요한 패치 개수 (360도 전체를 커버)
        num_patches = int(np.ceil(360.0 / step))
        # (3) yaw 중심 각 생성 (0 ~ 360 step 간격)
        yaw_centers = (np.arange(num_patches) * step) % 360.0
        # (4) e2p 함수 입력 범위로 변환 (-180 ~ 180)
        yaw_centers = np.where(yaw_centers > 180.0, yaw_centers - 360.0, yaw_centers)
        # (5) numpy array로 변환 (H2 x W4 x C)
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()

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
            )  # 반환: (H, W, C)
            t = torch.from_numpy(pers).permute(2, 0, 1)  # (C, H, W)
            patches.append(t)

        # (N, C, H, W) 형태로 병합 후, 모델 입력에 맞추기 위해 배치 차원 삽입
        stacked = torch.stack(patches, dim=0)    # (N, C, H, W)
        return stacked.unsqueeze(0)               # (1, N, C, H, W)

# -----------------------------------------------------------------------------------
# 4) 모델 및 프로세서 로드
# -----------------------------------------------------------------------------------
def get_model_and_processor(cfg: Dict[str, Any], device: torch.device):
    """
    설정 파일(cfg)을 참조하여, BLIP-2 또는 SurroundBlip 모델과 Processor를 생성 후 반환한다.
    """
    model_name    = cfg["model"]["name"]
    pretrain_path = cfg["model"]["pretrain_path"]

    # Processor 로드
    try:
        processor = Blip2Processor.from_pretrained(pretrain_path)
    except Exception as e:
        raise RuntimeError(f"Processor 로드 실패: {pretrain_path} | 오류: {e}")

    # 모델 로드 (Surround 또는 기본 BLIP-2 분기)
    if model_name.lower() == "surround":
        logger.info("[모델] SurroundBlip 로딩 시도")
        try:
            hf_config = Blip2Config.from_pretrained(pretrain_path)
            # 사용자 정의 num_query_tokens 오버라이드
            if "num_query_tokens" in cfg["model"]:
                hf_config.num_query_tokens = cfg["model"]["num_query_tokens"]
            # Q-Former 관련 설정 오버라이드
            if "qformer" in cfg["model"]:
                for key, value in cfg["model"]["qformer"].items():
                    if hasattr(hf_config.qformer_config, key):
                        setattr(hf_config.qformer_config, key, value)
            model = SurroundBlip.from_pretrained(
                pretrain_path,
                config=hf_config,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            raise RuntimeError(f"SurroundBlip 모델 로드 실패: {pretrain_path} | 오류: {e}")
    else:
        logger.info("[모델] BLIP-2 로딩 시도")
        try:
            model = Blip2ForConditionalGeneration.from_pretrained(pretrain_path)
        except Exception as e:
            raise RuntimeError(f"BLIP-2 모델 로드 실패: {pretrain_path} | 오류: {e}")

    model.to(device)
    model.eval()
    logger.info(f"[모델] '{model_name}' 로딩 완료, 디바이스: {device}")
    return processor, model

# -----------------------------------------------------------------------------------
# 5) 배치 단위 캡션 생성 함수
# -----------------------------------------------------------------------------------
def generate_captions(
    model: torch.nn.Module,
    processor: Blip2Processor,
    dataloader: DataLoader,
    gen_args: Dict[str, Any],
    device: torch.device
):
    """
    dataloader로부터 배치를 받아 모델로부터 예측 캡션을 생성하고,
    references, hypotheses, details를 리스트로 반환한다.
    """
    references:   List[List[str]]      = []
    hypotheses:   List[str]            = []
    details:      List[Dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pv = batch["pixel_values"]
        ids = batch["input_ids"]
        mask = batch["attention_mask"]

        # 모델 입력을 적절히 디바이스로 이동
        pv   = pv.to(device)
        ids  = ids.to(device)
        mask = mask.to(device)

        # 생성(Inference) 단계
        with torch.no_grad():
            try:
                gen_ids = model.generate(
                    pixel_values=pv,
                    input_ids=ids,
                    attention_mask=mask,
                    **gen_args
                )
            except Exception as e:
                logger.error(f"캡션 생성 중 오류 발생: {e}")
                batch_size = ids.size(0)
                gen_ids = torch.zeros((batch_size, 1), dtype=torch.long).to(device)

        # token ID → 문자열 디코딩
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

        # 결과 축적
        for url, query, ref, pred in zip(batch["url"], batch["query"], batch["annotation"], preds):
            references.append([ref])
            hypotheses.append(pred.strip())
            details.append({
                "url":        url,
                "query":      query,
                "reference":  ref,
                "hypothesis": pred.strip()
            })

    return references, hypotheses, details

# -----------------------------------------------------------------------------------
# 6) COCO-evalcap 지표 초기화 및 계산
# -----------------------------------------------------------------------------------
def get_scorers(metric_names: List[str]):
    """
    사용자 정의 metric_names 리스트에서 COCO-evalcap 계열 지표 객체만 생성하여 반환한다.
    CLIP-S, RefCLIP-S는 후처리 단계에서 별도로 계산한다.
    """
    scorers = []
    for m in metric_names:
        m_lower = m.lower()
        if m_lower.startswith("bleu"):
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        elif m_lower == "meteor":
            scorers.append((Meteor(), "METEOR"))
        elif m_lower in ["rouge", "rouge-l"]:
            scorers.append((Rouge(), "ROUGE_L"))
        elif m_lower == "cider":
            scorers.append((Cider(), "CIDEr"))
        elif m_lower == "spice":
            scorers.append((Spice(), "SPICE"))
        else:
            logger.info(f"get_scorers: '{m}' 은 COCO-evalcap 지표가 아니므로 건너뜁니다.")
    return scorers

def compute_scores(
    scorers: List[Any],
    references: List[List[str]],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    COCO-evalcap 형식 scorers를 이용하여, 전체 지표를 계산하고 'overall' 딕셔너리 형태로 반환한다.
    """
    gts = {i: references[i] for i in range(len(references))}
    res = {i: [hypotheses[i]] for i in range(len(hypotheses))}

    overall = {}
    for scorer, name in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
        except Exception as e:
            logger.error(f"{scorer.__class__.__name__} 지표 계산 중 오류 발생: {e}")
            continue

        if isinstance(name, list):
            for n, s in zip(name, score):
                overall[n] = float(s)
        else:
            overall[name] = float(score)

    return overall

# -----------------------------------------------------------------------------------
# 7) CLIP 모델 로딩 및 지표 계산 함수
# -----------------------------------------------------------------------------------
def get_clip_models(device: torch.device):
    """
    Hugging Face 허브에서 CLIP 모델과 프로세서를 로드하여 반환한다.
    """
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_processor  = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model      = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    return clip_model, clip_processor

def compute_clip_s(
    details: List[Dict[str, Any]],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device
) -> float:
    """
    details 리스트(각 원소에 'url'과 'hypothesis' 포함)를 순회하며,
    이미지와 생성문장 간 CLIP 유사도를 계산하여 평균을 반환한다.
    """
    similarities = []

    with torch.no_grad():
        for sample in tqdm(details, desc="Calculating CLIP-S"):
            # 1) 이미지 로딩
            try:
                image = Image.open(sample["url"]).convert("RGB")
            except Exception:
                # 이미지 로딩 실패 시 유사도 0으로 처리
                similarities.append(0.0)
                continue

            # 2) CLIP 프로세서로 전처리
            inputs = clip_processor(
                text=sample["hypothesis"],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            # 3) CLIP 모델에 입력하여 텍스트/이미지 임베딩 얻기
            outputs = clip_model(**inputs)
            img_emb  = outputs.image_embeds      # (1, D)
            txt_emb  = outputs.text_embeds       # (1, D)

            # 4) L2 정규화 후 코사인 유사도 계산
            img_emb_norm = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            txt_emb_norm = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
            sim = (img_emb_norm * txt_emb_norm).sum(dim=-1).item()  # 스칼라
            similarities.append(sim)

    if len(similarities) == 0:
        return 0.0
    return float(np.mean(similarities))

def compute_refclip_s(
    details: List[Dict[str, Any]],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device
) -> float:
    """
    details 리스트(각 원소에 'hypothesis'와 'reference' 포함)를 순회하며,
    생성문장과 정답문장 간 CLIP 텍스트 임베딩 기반 유사도를 계산하여 평균 반환.
    """
    text_sims = []

    with torch.no_grad():
        for sample in tqdm(details, desc="Calculating RefCLIP-S"):
            # 텍스트만 전처리 (두 문장을 한 번에 배치 처리)
            inputs = clip_processor(
                text=[sample["hypothesis"], sample["reference"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP 최대 토큰 길이
            ).to(device)

            # 두 문장 텍스트 임베딩 획득
            text_features = clip_model.get_text_features(**inputs)  # (2, D)
            h_emb = text_features[0].unsqueeze(0)  # (1, D)
            r_emb = text_features[1].unsqueeze(0)  # (1, D)

            # L2 정규화 후 코사인 유사도 계산
            h_norm = h_emb / h_emb.norm(p=2, dim=-1, keepdim=True)
            r_norm = r_emb / r_emb.norm(p=2, dim=-1, keepdim=True)
            sim = (h_norm * r_norm).sum(dim=-1).item()
            text_sims.append(sim)

    if len(text_sims) == 0:
        return 0.0
    return float(np.mean(text_sims))

# -----------------------------------------------------------------------------------
# 8) 결과 저장
# -----------------------------------------------------------------------------------
def save_results(
    overall: Dict[str, float],
    details: List[Dict[str, Any]],
    output_path: str
):
    """
    overall 지표와 샘플별 상세 정보를 JSON으로 저장한다.
    """
    out = {
        "overall": overall,
        "details": details
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise IOError(f"결과 저장 실패: {output_path} | 오류: {e}")
    logger.info(f"▶ 평가 완료. 결과가 저장되었습니다: {output_path}")

# -----------------------------------------------------------------------------------
# 9) 메인 함수
# -----------------------------------------------------------------------------------
def main():
    # 1) 인자 파싱
    parser = argparse.ArgumentParser(description="Evaluate BLIP-2 모델")
    parser.add_argument(
        "--config", type=str, required=True,
        help="평가용 YAML 파일 경로 (예: config/eval.yaml)"
    )
    args = parser.parse_args()

    # 2) 구성 파일 로드 및 검증
    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error(f"config 로드 오류: {e}")
        return

    # 3) 디바이스 설정
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"[디바이스] {device}")

    # 4) 모델 및 프로세서 로드
    try:
        processor, model = get_model_and_processor(cfg, device)
    except Exception as e:
        logger.error(f"모델/프로세서 로드 오류: {e}")
        return

    # 5) 데이터셋 및 DataLoader 준비
    data_dir  = cfg["data"]["dir"]
    test_file = cfg["data"]["test_file"]
    csv_path  = os.path.join(data_dir, test_file)

    try:
        dataset = EvalDataset(
            csv_file=csv_path,
            processor=processor,
            max_length=cfg["data"].get("max_length", None),
            image_size=cfg["data"].get("image_size", [224, 224]),
            do_crop=cfg["data"].get("do_crop", False),
            fov=cfg["data"].get("fov", None),
            overlap_ratio=cfg["data"].get("overlap_ratio", None)
        )
    except Exception as e:
        logger.error(f"데이터셋 초기화 오류: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["eval"]["num_workers"],
        shuffle=False,
        pin_memory=True
    )

    # 6) 생성 파라미터
    gen_args = {
        "max_new_tokens": cfg["generate"]["max_length"],
        "num_beams": cfg["generate"]["num_beams"]
    }

    # 7) COCO-evalcap 지표 초기화
    metric_names = cfg["eval"].get(
        "metrics",
        ["BLEU", "METEOR", "ROUGE", "CIDEr", "SPICE"]
    )
    scorers = get_scorers(metric_names)

    # 8) 캡션 생성
    references, hypotheses, details = generate_captions(
        model=model,
        processor=processor,
        dataloader=dataloader,
        gen_args=gen_args,
        device=device
    )

    # 9) COCO-evalcap 지표 계산
    overall_scores = compute_scores(scorers, references, hypotheses)
    logger.info(f"[지표] COCO-evalcap 결과: {overall_scores}")

    # 10) CLIP 기반 지표 후처리
    # "CLIP-S", "RefCLIP-S" 가 metric_names에 있으면 각각 계산
    clip_model = None
    clip_processor = None

    if "clip-s" in [m.lower() for m in metric_names] or \
       "refclip-s" in [m.lower() for m in metric_names]:
        # CLIP 모델과 프로세서 로드
        clip_model, clip_processor = get_clip_models(device)

    if "clip-s" in [m.lower() for m in metric_names]:
        clip_s = compute_clip_s(details, clip_model, clip_processor, device)
        overall_scores["CLIP-S"] = clip_s
        logger.info(f"[지표] CLIP-S 결과: {clip_s:.4f}")

    if "refclip-s" in [m.lower() for m in metric_names]:
        refclip_s = compute_refclip_s(details, clip_model, clip_processor, device)
        overall_scores["RefCLIP-S"] = refclip_s
        logger.info(f"[지표] RefCLIP-S 결과: {refclip_s:.4f}")

    # 11) 결과 저장
    output_path = cfg["output"]["result_file"]
    try:
        save_results(overall_scores, details, output_path)
    except Exception as e:
        logger.error(f"결과 저장 오류: {e}")

if __name__ == "__main__":
    main()