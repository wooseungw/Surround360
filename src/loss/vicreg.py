import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """VICReg (Variance-Invariance-Covariance Regularization) 변형 손실 함수.
    
    겹치는 이미지 영역(좌-우측 절반, 우-좌측 절반)의 특징 일관성을 
    강화하기 위한 손실 함수입니다.
    """
    
    def __init__(self, 
                 sim_coef=25.0, 
                 var_coef=25.0, 
                 cov_coef=1.0, 
                 eps=1e-4):
        """
        Args:
            sim_coef: 유사성(invariance) 손실 계수
            var_coef: 분산(variance) 손실 계수
            cov_coef: 공분산(covariance) 손실 계수
            eps: 수치 안정성을 위한 작은 상수
        """
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.eps = eps
    
    def forward(self, left_embeds, right_embeds):
        """
        겹치는 영역의 특징 간 VICReg 손실을 계산합니다.
        
        Args:
            left_embeds: 좌측 이미지 특징 (B, N, D) 또는 (N, D)
            right_embeds: 우측 이미지 특징 (B, N, D) 또는 (N, D)
            
        Returns:
            total_loss: 전체 손실
            losses: 개별 손실 컴포넌트 (dictionary)
        """
        # 입력 차원 확인 및 처리
        if left_embeds.dim() == 2:
            left_embeds = left_embeds.unsqueeze(0)
            right_embeds = right_embeds.unsqueeze(0)
            
        batch_size = left_embeds.size(0)
        
        # 1. 유사성(invariance) 손실 - 겹치는 영역이 유사해야 함
        sim_loss = F.mse_loss(left_embeds, right_embeds)
        
        # 2. 분산(variance) 손실 - 각 차원이 충분한 분산을 가져야 함
        # 배치 내 각 샘플에 대해 평균 제거
        left_centered = left_embeds - left_embeds.mean(dim=1, keepdim=True)
        right_centered = right_embeds - right_embeds.mean(dim=1, keepdim=True)
        
        # 표준 편차 계산 (배치 내 각 특징별)
        std_left = torch.sqrt(left_centered.var(dim=1) + self.eps)
        std_right = torch.sqrt(right_centered.var(dim=1) + self.eps)
        
        # 분산 손실: 각 차원의 표준 편차가 1보다 작으면 페널티
        var_loss = torch.mean(F.relu(1 - std_left)) + torch.mean(F.relu(1 - std_right))
        
        # 3. 공분산(covariance) 손실 - 특징 간 상관관계 최소화
        # 각 배치 샘플에 대해 공분산 행렬 계산
        left_flat = left_centered.flatten(1)  # (B, N*D)
        right_flat = right_centered.flatten(1)  # (B, N*D)
        
        # 배치 크기가 1이면 특별 처리
        if batch_size == 1:
            # 공분산 계산 불가능하므로 특징 차원 간 상관관계만 고려
            d = left_flat.size(1)
            left_cov = (left_flat.T @ left_flat) / max(1, d - 1)
            right_cov = (right_flat.T @ right_flat) / max(1, d - 1)
        else:
            # 배치 차원에서 공분산 계산
            left_cov = (left_flat.T @ left_flat) / max(1, batch_size - 1)
            right_cov = (right_flat.T @ right_flat) / max(1, batch_size - 1)
        
        # 대각선 마스크 (대각 요소 제외)
        diag = torch.eye(left_cov.size(0), device=left_cov.device)
        cov_loss = ((left_cov * (1 - diag)).pow(2).sum() + 
                    (right_cov * (1 - diag)).pow(2).sum()) / 2
        
        # 전체 손실 계산
        total_loss = self.sim_coef * sim_loss + self.var_coef * var_loss + self.cov_coef * cov_loss
        
        return total_loss, {
            "sim_loss": sim_loss.detach(), 
            "var_loss": var_loss.detach(), 
            "cov_loss": cov_loss.detach()
        }