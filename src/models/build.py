from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from .config import VisionLanguageConfig
from .projector import build_vision_projector
from .newline import NewlineTokenInserter
# from .resampler.mamba_ssm.modules.mamba_compressor import MambaCompressor
from .constant import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class CustomVLMModel(PreTrainedModel):
    """Vision–Language 모델 (Vision Encoder + Projector + LLM)."""

    config_class = VisionLanguageConfig

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: VisionLanguageConfig,
        tokenizer: AutoTokenizer,
        vision_dtype: torch.dtype = torch.float16, 
        llm_dtype: torch.dtype = torch.float16,
        **kwargs,
        ) -> None:
        super().__init__(config, **kwargs)
        self.config = config
        self.vision_dtype = vision_dtype
        self.llm_dtype    = llm_dtype
        # 1) Vision encoder --------------------------------------------------
        vision_model = AutoModel.from_pretrained(
            config.vision_model_name,
            torch_dtype=vision_dtype,
        ).to(vision_dtype)
        # Check if vision_model attribute exists
        if hasattr(vision_model, 'vision_model'):
            self.vision_encoder = vision_model.vision_model
        else:
            self.vision_encoder = vision_model
        d_v = self.config.vision_config.hidden_size
        
        # 6) Language model -------------------------------------------------
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.language_model_name,
            torch_dtype=llm_dtype,
        ).to(llm_dtype)
        d_l = self.config.language_config.hidden_size
        # 2) Projector -------------------------------------------------------
        self.projector = build_vision_projector(
            d_v=d_v,
            d_l=d_l,
            projector_type=config.projector_type,
            vision_cfg=self.config.vision_config,
        ).to(vision_dtype)
        # 3) Optional resampler ---------------------------------------------
        if getattr(config, "use_resampler", False):
            from src.models.resampler.mamba_ssm.modules.mamba_compressor import MambaCompressor
            self.resampler = MambaCompressor(d_model=d_l, n_layer=1, fp32=False)
        else:
            self.resampler = None
        print(f"Vision encoder: {self.vision_encoder.__class__.__name__}","LLM: ", self.llm.__class__.__name__)
        # 4) NEWLINE token parameter ----------------------------------------
        self.image_newline = nn.Parameter(torch.zeros(d_l, dtype=vision_dtype))
        self.newline_inserter = NewlineTokenInserter(config)

        # 5) Tokenizer & special tokens -------------------------------------
        self.tokenizer = tokenizer
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right"
        # self._register_special_tokens()
        # 7) Optional freezing ----------------------------------------------
        if getattr(config, "freeze_vision", True):
            self._freeze(self.vision_encoder)
        if getattr(config, "freeze_llm", False):
            self._freeze(self.llm)

        self.post_init()
        # Let HF Trainer know this wrapper can toggle gradient checkpointing
        self.supports_gradient_checkpointing = True

    # ------------------------------------------------------------------ #
    # SPECIAL‑TOKEN HANDLING
    # ------------------------------------------------------------------ #
    # def _register_special_tokens(self):
    #     """LLaVA 스타일의 특수 토큰 등록 및 커스텀 ID 지정"""
    #     # 1. 기존 토크나이저 크기 확인
    #     base_vocab_size = len(self.tokenizer)
    #     # 3. 추가할 특수 토큰 정의
    #     special_tokens = {"additional_special_tokens": [
    #         DEFAULT_IMAGE_TOKEN,
    #         DEFAULT_IMAGE_PATCH_TOKEN, 
    #         DEFAULT_IM_START_TOKEN,
    #         DEFAULT_IM_END_TOKEN,
    #     ]}
        
    #     # 4. 토큰 추가
    #     added = self.tokenizer.add_special_tokens(special_tokens)
    #     # print(f"추가된 토큰 수: {added}")
        
    #     # 5. 토크나이저 타입 확인 (tokenizer.__class__.__name__으로 출력)
    #     # print(f"토크나이저 타입: {self.tokenizer.__class__.__name__}")
        
    #     # 6. 토큰-ID 매핑 설정 (BPE 기반 토크나이저 예시)
    #     if hasattr(self.tokenizer, 'encoder') and hasattr(self.tokenizer, 'decoder'):
    #         # 원하는 매핑 정의 - 여기서는 custom_id를 사용자가 설정
    #         custom_id_map = {
    #             DEFAULT_IMAGE_TOKEN: base_vocab_size,      # base_vocab_size부터 순차적으로 할당
    #             DEFAULT_IMAGE_PATCH_TOKEN: base_vocab_size + 1,
    #             DEFAULT_IM_START_TOKEN: base_vocab_size + 2, 
    #             DEFAULT_IM_END_TOKEN: base_vocab_size + 3,
    #         }
            
    #         # 매핑 설정
    #         for token, custom_id in custom_id_map.items():
    #             # 자동 할당된 ID 확인
    #             auto_id = self.tokenizer.convert_tokens_to_ids(token)
    #             print(f"토큰 '{token}' - 자동 할당 ID: {auto_id}")
                
    #             # 자동 할당된 ID 제거
    #             if token in self.tokenizer.encoder:
    #                 del self.tokenizer.encoder[token]
    #             if auto_id in self.tokenizer.decoder:
    #                 del self.tokenizer.decoder[auto_id]
                
    #             # 커스텀 ID 설정
    #             self.tokenizer.encoder[token] = custom_id
    #             self.tokenizer.decoder[custom_id] = token
                
    #             # 추가 토큰 매핑 수정
    #             if hasattr(self.tokenizer, 'added_tokens_encoder'):
    #                 self.tokenizer.added_tokens_encoder[token] = custom_id
    #             if hasattr(self.tokenizer, 'added_tokens_decoder'):
    #                 self.tokenizer.added_tokens_decoder[custom_id] = token
                
    #             print(f"토큰 '{token}' - 설정된 커스텀 ID: {custom_id}")
    #     else:
    #         print("경고: 현재 토크나이저는 직접 ID 매핑을 지원하지 않습니다.")
        
    #     # 7. 모델 임베딩 크기 조정
    #     self.llm.resize_token_embeddings(len(self.tokenizer))
        
    #     # 8. 결과 확인
    #     print(f"추가 후 토큰 수: {len(self.tokenizer)}")
    #     print(f"추가 후 특수 토큰 목록: {self.tokenizer.all_special_tokens}")
    #     for token in special_tokens["additional_special_tokens"]:
    #         curr_id = self.tokenizer.convert_tokens_to_ids(token)
    #         print(f"토큰: {token}, ID: {curr_id}")
        
    #     # # 9. 내부 처리에서의 특수 인덱스 사용 설명
    #     # print("\n참고: 토크나이저에서는 음수 ID를 직접 사용할 수 없습니다.")
    #     # print("음수 값(IGNORE_INDEX=-100, IMAGE_TOKEN_INDEX=-200)은 내부 처리 로직에서 특별한 목적으로 사용됩니다.")
    # ------------------------------------------------------------------ #
    # UTILS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # GRADIENT CHECKPOINTING TOGGLE (delegate to LLM)
    # ------------------------------------------------------------------ #
    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the underlying language model.
        HF Trainer expects this method to exist when gradient_checkpointing=True.
        """
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(**kwargs)
        # Disable cache to save memory
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        # Re‑enable cache if available
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = True

    # ------------------------------------------------------------------ #
    # VISION → PROJECTOR
    # ------------------------------------------------------------------ #
    def _get_vision_embeds(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Vision encoder → (optional) resampler → projector → LLM‑space embeds.
        Handles both 4D (B, C, H, W) and 5D (B, T, C, H, W) inputs.
        """
        # Check if input is 5D (batch, frames, channels, height, width)
        is_video = pixel_values.dim() == 5
        
        if is_video:
            B, T, C, H, W = pixel_values.shape
            # Reshape to (B*T, C, H, W) for vision encoder
            pixel_values = pixel_values.view(B * T, C, H, W)
        
        # Process through vision encoder
        v_feat = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  # (B*T, N, d_v)
        v_emb = self.projector(v_feat)  # (B*T, N', d_l)
        
        return v_emb  # (B*T, N', d_l)
    
    def _get_2dPool(self, features: torch.FloatTensor, stride: int = 2):
        """ViT patch token을 2‑D grid 로 reshape 후 pooling."""
        # feature: (B, N, d_v)
        
        num_frames, num_tokens, num_dim = features.shape
        height = weight = int(math.sqrt(num_tokens))
        
        # CLS 토큰 처리 - 첫 번째 토큰은 제외
        has_cls_token = (num_tokens % 2 == 1)  # 14x14 + 1 = 197
        if has_cls_token:
            # cls_token = features[:, 0:1, :]  # CLS 토큰 분리
            features = features[:, 1:, :]
        
        features = features.view(num_frames, height, weight, num_dim)
        # print(f"feature 모양: {features.shape}")  # 디버깅
        if self.config.use_resampler:
            space_time_tokens = features.unsqueeze(0)
            
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, d_v, H, W)
        # print(f"feature 모양: {features.shape}")  # 디버깅
        if self.config.mm_spatial_pool_mode == "average":
            features = nn.functional.avg_pool2d(features, stride) 
        elif self.config.mm_spatial_pool_mode == "max":
            features = nn.functional.max_pool2d(features, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = features.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            features = nn.functional.interpolate(features, size=scaled_shape, mode='bilinear') 
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        # print(f"pooling 후 feature 모양: {features.shape}")  # 디버깅
        # (64, 3584, H, W) -> (64, 3584, H//stride, W//stride)
        features = features.permute(0, 2, 3, 1)
        # (64, H//stride, W//stride, 3584)
        features = features.view(num_frames, -1, num_dim)
        # print(f"feature 모양: {features.shape}")  # 디버깅
        if self.config.use_resampler:
            # print("resampler 사용")
            features = features.unsqueeze(0)
            # print(f"space_time_tokens 모양: {space_time_tokens.shape}")  # 디버깅
            # print(f"feature.unsqueeze 모양: {features.shape}")  # 디버깅
            features = self.resampler(space_time_tokens, features)
            features = torch.squeeze(features, 0)
            # print(f"feature 모양: {features.shape}")  # 디버깅
        # print(f"feature 모양: {features.shape}")  # 디버깅
        return features.to(self.llm_dtype)  # (B, H//stride * W//stride, d_l)
        
    # ------------------------------------------------------------------ #
    # IMAGE TOKEN REPLACEMENT (re‑implemented)
    # ------------------------------------------------------------------ #
    def preprocess_image_tokens(self, input_ids, attention_mask=None):
        """토크나이저 결과에서 <image> 토큰 ID를 IMAGE_TOKEN_INDEX(-200)로 변환"""
        # 토크나이저에서 <image> 토큰의 실제 ID 가져오기
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        
        # 입력 ID 복사본 만들기 (변경 가능하도록)
        processed_input_ids = input_ids.clone()
        
        # <image> 토큰 ID를 IMAGE_TOKEN_INDEX로 변경
        mask = (processed_input_ids == image_token_id)
        processed_input_ids[mask] = IMAGE_TOKEN_INDEX
        
        return processed_input_ids
    def _replace_image_tokens_with_features(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        image_features: list,   # (B, M, d_l)
        embed_tokens_fn: nn.Module,
        image_token_index: int,
        *,
        ignore_index: int = IGNORE_INDEX,
        max_length: int = None,
        padding_side: str = "right",
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """<image> 토큰을 실제 vis embedding 으로 치환.
        - 배치마다 image_features[b] 를 사용
        - 텍스트 세그먼트가 길이 0 이더라도 차원 일치 보장
        - 이미지 토큰이 없으면 prefix 전략 미사용 (plain 텍스트)"""

        device = input_ids.device
        B = input_ids.size(0)
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
        else:
            attention_mask = attention_mask.bool()

        # pad 제거
        ids_list = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        lbl_list = [lbl[mask] for lbl, mask in zip(labels, attention_mask)]

        seq_embeds, seq_labels = [], []
        for b in range(B):
            ids = ids_list[b]
            lbls = lbl_list[b]
            vis_emb = image_features[b]                   # (M, d
            if vis_emb.dim() == 1:
                vis_emb = vis_emb.unsqueeze(0)

            img_pos = (ids == image_token_index).nonzero(as_tuple=False).flatten()
            if img_pos.numel() == 0:
                # 텍스트만
                seq_embeds.append(embed_tokens_fn(ids))
                seq_labels.append(lbls)
                continue

            # split points 포함해 세그먼트 추출
            split_pts = torch.cat([torch.tensor([-1], device=device), img_pos, torch.tensor([ids.size(0)], device=device)])
            seg_emb, seg_lbl = [], []
            for i in range(split_pts.numel() - 1):
                s = split_pts[i] + 1
                e = split_pts[i + 1]
                txt_ids = ids[s:e]
                txt_lbl = lbls[s:e]
                if txt_ids.numel() > 0:
                    txt_emb = embed_tokens_fn(txt_ids)
                else:
                    # 빈 세그먼트: (0, d)
                    txt_emb = vis_emb[:0]
                seg_emb.append(txt_emb)
                seg_lbl.append(txt_lbl)

                # 이미지 토큰 위치면 vis_emb 삽입
                if i < img_pos.numel():
                    seg_emb.append(vis_emb)
                    seg_lbl.append(torch.full((vis_emb.size(0),), ignore_index, dtype=lbls.dtype, device=device))

            seq_embeds.append(torch.cat(seg_emb, dim=0))
            seq_labels.append(torch.cat(seg_lbl, dim=0))

        # -------------------------------- padding
        if max_length is not None:
            seq_embeds = [e[:max_length] for e in seq_embeds]
            seq_labels = [l[:max_length] for l in seq_labels]

        L = max(e.size(0) for e in seq_embeds)
        D = seq_embeds[0].size(1)
        pad_emb = lambda n: torch.zeros((n, D), dtype=seq_embeds[0].dtype, device=device)

        pad_embeds, pad_labels = [], torch.full((B, L), ignore_index, dtype=seq_labels[0].dtype, device=device)
        pad_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        pos_ids = torch.zeros((B, L), dtype=torch.long, device=device)

        for i, (emb, lab) in enumerate(zip(seq_embeds, seq_labels)):
            cur = emb.size(0)
            if padding_side == "left":
                pad = pad_emb(L - cur)
                emb = torch.cat([pad, emb], dim=0)
                pad_labels[i, -cur:] = lab
                pad_mask[i, -cur:] = 1
                pos_ids[i, -cur:] = torch.arange(cur, device=device)
            else:
                pad = pad_emb(L - cur)
                emb = torch.cat([emb, pad], dim=0)
                pad_labels[i, :cur] = lab
                pad_mask[i, :cur] = 1
                pos_ids[i, :cur] = torch.arange(cur, device=device)
            pad_embeds.append(emb)

        pad_embeds = torch.stack(pad_embeds, dim=0)
        return pad_embeds, pad_labels, pad_mask, pos_ids

    def _prepare_multimodal_inputs(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ):
        """forward와 generate에서 공통적으로 사용되는 입력 전처리 로직"""
        # 1. 토큰 전처리
        processed_input_ids = self.preprocess_image_tokens(input_ids)
        if labels is not None:
            processed_labels = self.preprocess_image_tokens(labels)
        else:
            processed_labels = processed_input_ids.clone()
        
        # 2. 비전 인코딩
        self.current_batch_size = processed_input_ids.size(0)
        v_embs = self._get_vision_embeds(pixel_values)  # (B*num_samples, N', d_l)
        
        # 3. v_embs 배치 처리
        v_embs = list(torch.split(v_embs, self.current_batch_size, dim=0))  # (B, num_samples, N', d_l)
        
        for i, v_emb in enumerate(v_embs):
            # 4. 비전 임베딩 풀링
            if self.config.mm_spatial_pool_mode != "none":
                v_emb = self._get_2dPool(v_emb, stride=2)
            # 5. 줄바꿈 토큰 삽입
            v_emb = self.newline_inserter(v_emb, self.image_newline)
            v_embs[i] = v_emb
        
        # 6. 이미지 토큰 대체
        inp_emb, pad_lbl, pad_mask, pos_ids = self._replace_image_tokens_with_features(
            input_ids=processed_input_ids,
            labels=processed_labels,
            attention_mask=attention_mask,
            image_features=v_embs,
            embed_tokens_fn=self.llm.get_input_embeddings(),
            image_token_index=IMAGE_TOKEN_INDEX,
            ignore_index=IGNORE_INDEX,
            max_length=self.config.language_config.max_position_embeddings,
            padding_side=self.tokenizer.padding_side,
        )
        
        return inp_emb, pad_lbl, pad_mask, pos_ids

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 공통 입력 전처리 메서드 호출
        inp_emb, pad_lbl, pad_mask, pos_ids = self._prepare_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        inp_emb.requires_grad_()
        # LLM 모델 호출
        return self.llm(
            inputs_embeds=inp_emb,
            attention_mask=pad_mask if attention_mask is not None else None,
            position_ids=pos_ids,
            labels=pad_lbl,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):
        # 공통 입력 전처리 메서드 호출
        inp_emb, _, pad_mask, pos_ids = self._prepare_multimodal_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # generate에 필요한 model_kwargs 구성
        model_kwargs = {
            "inputs_embeds": inp_emb,
            "attention_mask": pad_mask if attention_mask is not None else None,
            "position_ids": pos_ids,
        }
        
        # 생성 수행
        outputs = self.llm.generate(
            input_ids=None,  # input_ids 대신 inputs_embeds 사용
            **model_kwargs,
            **generate_kwargs
        )
        return outputs
if __name__ == "__main__":
    vision = ["facebook/dino-vitb16"]
    cfg = VisionLanguageConfig(
        vision_model_name=vision[0],
        language_model_name="gpt2",
        use_resampler=True,
        mm_spatial_pool_mode= "average",
    )
    model = CustomVLMModel(cfg).eval().to("cuda")

    dummy_img = torch.randn(4, 3, 224, 224, device="cuda", dtype=torch.float16)
    prompt = f"{DEFAULT_IM_START_TOKEN} hello {DEFAULT_IMAGE_TOKEN} world {DEFAULT_IM_END_TOKEN}"
    # Create a batch of prompts
    prompts = [
        f"{DEFAULT_IM_START_TOKEN} hello {DEFAULT_IMAGE_TOKEN} world {DEFAULT_IM_END_TOKEN}",
        f"{DEFAULT_IM_START_TOKEN} test {DEFAULT_IMAGE_TOKEN} example {DEFAULT_IM_END_TOKEN}"
    ]
    tok = model.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        out = model(pixel_values=dummy_img, input_ids=tok.input_ids, attention_mask=tok.attention_mask, labels=tok.input_ids)
        print("Logits shape:", out.logits.shape)
    with torch.no_grad():
        out = model.generate(pixel_values=dummy_img, input_ids=tok.input_ids,max_new_tokens =50 ,attention_mask=tok.attention_mask, labels=tok.input_ids)
        print("Generated IDs:", out)
        out = model.tokenizer.batch_decode(out, skip_special_tokens=True)
        print("Generated text:", out)

    
