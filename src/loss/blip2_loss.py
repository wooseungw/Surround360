import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

class QFormerLosses(nn.Module):
    """
    Q-Former 학습을 위한 3가지 손실 함수 구현:
    1. Image-Text Contrastive Learning (ITC)
    2. Image-grounded Text Generation (ITG) 
    3. Image-Text Matching (ITM)
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 hard_negative_ratio: float = 0.5,
                 itc_weight: float = 1.0,
                 itg_weight: float = 1.0,
                 itm_weight: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.hard_negative_ratio = hard_negative_ratio
        self.itc_weight = itc_weight
        self.itg_weight = itg_weight
        self.itm_weight = itm_weight
        
        # ITM 분류를 위한 헤드 (쿼리 차원 -> 2)
        self.itm_head = None  # 모델에서 설정됨
    
    def compute_itc_loss(self, 
                        query_outputs: torch.Tensor,  # (B, Q, D)
                        text_features: torch.Tensor,  # (B, D)
                        ) -> torch.Tensor:
        """
        Image-Text Contrastive Learning 손실 계산
        
        Args:
            query_outputs: Q-Former의 쿼리 출력 (B, 32, D)
            text_features: 텍스트 [CLS] 토큰 특징 (B, D)
        """
        batch_size = query_outputs.size(0)
        
        # 쿼리 출력을 평균 풀링하여 이미지 표현 생성
        image_features = query_outputs.mean(dim=1)  # (B, D)
        
        # L2 정규화
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 유사도 행렬 계산
        sim_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 대각선이 positive pairs
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 양방향 대조 손실
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)
        
        itc_loss = (loss_i2t + loss_t2i) / 2
        return itc_loss
    
    def compute_itg_loss(self,
                        model,
                        query_outputs: torch.Tensor,  # (B, Q, D)
                        input_ids: torch.Tensor,      # (B, L)
                        attention_mask: torch.Tensor, # (B, L)
                        labels: torch.Tensor,         # (B, L)
                        ) -> torch.Tensor:
        """
        Image-grounded Text Generation 손실 계산
        
        Args:
            model: Q-Former 모델
            query_outputs: Q-Former의 쿼리 출력
            input_ids: 텍스트 토큰 ID
            attention_mask: 텍스트 어텐션 마스크
            labels: 생성 타겟 라벨
        """
        batch_size, num_queries, hidden_size = query_outputs.shape
        seq_len = input_ids.size(1)
        
        # 텍스트 임베딩 생성
        text_embeddings = model.embeddings(input_ids)  # (B, L, D)
        
        # 쿼리와 텍스트를 결합
        combined_embeddings = torch.cat([query_outputs, text_embeddings], dim=1)  # (B, Q+L, D)
        
        # Causal attention mask 생성 (쿼리는 모든 토큰에, 텍스트는 이전 토큰들에만 접근)
        total_len = num_queries + seq_len
        causal_mask = torch.tril(torch.ones(total_len, total_len)).bool()
        
        # 쿼리 부분은 모든 위치에 접근 가능
        causal_mask[:num_queries, :] = True
        causal_mask = causal_mask.to(query_outputs.device)
        
        # 확장된 어텐션 마스크
        extended_attention_mask = torch.cat([
            torch.ones(batch_size, num_queries, device=query_outputs.device),
            attention_mask.float()
        ], dim=1)
        
        # Q-Former forward pass with causal masking
        outputs = model.encoder(
            hidden_states=combined_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            query_length=num_queries,
        )
        
        # 텍스트 부분만 추출하여 언어 모델링 손실 계산
        text_hidden_states = outputs.last_hidden_state[:, num_queries:, :]  # (B, L, D)
        
        # 언어 모델 헤드를 통과 (여기서는 간단히 linear projection으로 가정)
        vocab_size = model.config.vocab_size
        lm_logits = F.linear(text_hidden_states, model.embeddings.word_embeddings.weight)
        
        # Shift labels for next token prediction
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 손실 계산 (패딩 토큰 제외)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        itg_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        return itg_loss
    
    def compute_itm_loss(self,
                        model,
                        query_outputs: torch.Tensor,     # (B, Q, D)
                        text_embeddings: torch.Tensor,   # (B, L, D)
                        attention_mask: torch.Tensor,    # (B, L)
                        itm_labels: torch.Tensor,        # (B,) 1=match, 0=no-match
                        ) -> torch.Tensor:
        """
        Image-Text Matching 손실 계산
        
        Args:
            model: Q-Former 모델
            query_outputs: Q-Former의 쿼리 출력
            text_embeddings: 텍스트 임베딩
            attention_mask: 텍스트 어텐션 마스크
            itm_labels: 매칭 라벨 (1=일치, 0=불일치)
        """
        batch_size, num_queries, hidden_size = query_outputs.shape
        seq_len = text_embeddings.size(1)
        
        # 쿼리와 텍스트를 결합
        combined_embeddings = torch.cat([query_outputs, text_embeddings], dim=1)
        
        # Bi-directional attention mask (양방향 상호작용 허용)
        extended_attention_mask = torch.cat([
            torch.ones(batch_size, num_queries, device=query_outputs.device),
            attention_mask.float()
        ], dim=1)
        
        # Q-Former forward pass with bi-directional attention
        outputs = model.encoder(
            hidden_states=combined_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            query_length=num_queries,
        )
        
        # 첫 번째 쿼리 토큰을 [CLS]로 사용
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, D)
        
        # ITM 분류 헤드
        if self.itm_head is None:
            self.itm_head = nn.Linear(hidden_size, 2).to(query_outputs.device)
        
        itm_logits = self.itm_head(cls_output)  # (B, 2)
        
        # 이진 분류 손실
        itm_loss = F.cross_entropy(itm_logits, itm_labels)
        
        return itm_loss
    
    def create_hard_negatives(self, 
                             image_features: torch.Tensor,
                             text_features: torch.Tensor,
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard negative mining을 위한 부정적 쌍 생성
        
        Args:
            image_features: 이미지 특징 (B, D)
            text_features: 텍스트 특징 (B, D)
            
        Returns:
            hard_neg_images: Hard negative 이미지들
            hard_neg_texts: Hard negative 텍스트들
        """
        batch_size = image_features.size(0)
        
        # 유사도 행렬 계산
        sim_matrix = torch.matmul(
            F.normalize(image_features, dim=-1),
            F.normalize(text_features, dim=-1).T
        )
        
        # 대각선 마스킹 (positive pairs 제외)
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # 가장 유사한 negative pairs 선택
        num_hard_negs = int(batch_size * self.hard_negative_ratio)
        
        # 이미지 -> 텍스트 hard negatives
        _, hard_neg_text_indices = sim_matrix.topk(k=1, dim=1)
        hard_neg_text_indices = hard_neg_text_indices.squeeze(1)
        
        # 텍스트 -> 이미지 hard negatives  
        _, hard_neg_image_indices = sim_matrix.T.topk(k=1, dim=1)
        hard_neg_image_indices = hard_neg_image_indices.squeeze(1)
        
        return hard_neg_image_indices[:num_hard_negs], hard_neg_text_indices[:num_hard_negs]
    
    def forward(self,
                model,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Q-Former의 3가지 손실을 모두 계산
        
        Args:
            model: SurroundBlip 모델
            pixel_values: 입력 이미지 (B, P, C, H, W)
            input_ids: 텍스트 토큰 ID (B, L)
            attention_mask: 텍스트 어텐션 마스크 (B, L)
            labels: 생성 타겟 (B, L), ITG용
            
        Returns:
            손실 딕셔너리
        """
        batch_size = pixel_values.size(0)
        
        # 1. 이미지 인코딩 및 Q-Former 처리
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        
        # Vision encoder
        vision_outputs = model.vision_model(pixel_values_flat)
        image_embeds = vision_outputs.last_hidden_state  # (B*P, S, D)
        
        # Reshape for multi-view
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        image_embeds = image_embeds.view(B, P * S, D)
        
        # Q-Former processing
        image_attention_mask = torch.ones((B, P * S), dtype=torch.long, device=pixel_values.device)
        query_tokens = model.query_tokens.expand(B, -1, -1)
        
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_features = query_outputs.last_hidden_state  # (B, Q, D)
        
        # 2. 텍스트 인코딩
        text_embeddings = model.qformer.embeddings(input_ids)  # (B, L, D)
        
        # [CLS] 토큰 특징 (ITC용)
        text_features = text_embeddings[:, 0, :]  # (B, D)
        
        losses = {}
        
        # 3. ITC Loss
        itc_loss = self.compute_itc_loss(query_features, text_features)
        losses['itc_loss'] = itc_loss * self.itc_weight
        
        # 4. ITG Loss (labels가 제공된 경우)
        if labels is not None:
            itg_loss = self.compute_itg_loss(
                model.qformer, query_features, input_ids, attention_mask, labels
            )
            losses['itg_loss'] = itg_loss * self.itg_weight
        
        # 5. ITM Loss
        # Hard negative sampling으로 부정적 쌍 생성
        image_feats_pooled = query_features.mean(dim=1)  # (B, D)
        hard_neg_img_idx, hard_neg_txt_idx = self.create_hard_negatives(
            image_feats_pooled, text_features
        )
        
        # Positive pairs
        pos_query_features = query_features
        pos_text_embeddings = text_embeddings
        pos_attention_mask = attention_mask
        pos_labels = torch.ones(batch_size, dtype=torch.long, device=pixel_values.device)
        
        # Negative pairs (일부만 hard negatives로 교체)
        if len(hard_neg_img_idx) > 0 and len(hard_neg_txt_idx) > 0:
            neg_query_features = query_features[hard_neg_img_idx]
            neg_text_embeddings = text_embeddings[hard_neg_txt_idx]
            neg_attention_mask = attention_mask[hard_neg_txt_idx]
            neg_labels = torch.zeros(len(hard_neg_img_idx), dtype=torch.long, device=pixel_values.device)
            
            # Combine positive and negative pairs
            all_query_features = torch.cat([pos_query_features, neg_query_features], dim=0)
            all_text_embeddings = torch.cat([pos_text_embeddings, neg_text_embeddings], dim=0)
            all_attention_mask = torch.cat([pos_attention_mask, neg_attention_mask], dim=0)
            all_itm_labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            all_query_features = pos_query_features
            all_text_embeddings = pos_text_embeddings
            all_attention_mask = pos_attention_mask
            all_itm_labels = pos_labels
        
        itm_loss = self.compute_itm_loss(
            model.qformer, all_query_features, all_text_embeddings, 
            all_attention_mask, all_itm_labels
        )
        losses['itm_loss'] = itm_loss * self.itm_weight
        
        # 6. 총 손실
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses