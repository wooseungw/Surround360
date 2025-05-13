import torch.nn.functional as F
from torch import nn



class VICRegLoss(nn.Module):
    """VICReg (Variance-Invariance-Covariance Regularization) 변형 손실 함수.
    
    50% 겹치는 이미지 영역(좌-우측 절반, 우-좌측 절반)의 특징 일관성을 
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
            left_embeds: 좌측 이미지 특징 (B, N, D)
            right_embeds: 우측 이미지 특징 (B, N, D)
            
        Returns:
            total_loss: 전체 손실
            losses: 개별 손실 컴포넌트 (dictionary)
        """
        # 패치 수 파악 및 겹치는 영역 추출
        B, N, D = left_embeds.shape
        n_half = N // 2
        
        # 좌측 이미지의 우측 절반과 우측 이미지의 좌측 절반 추출
        left_right_half = left_embeds[:, n_half:, :]   # 좌측 이미지 우측 절반
        right_left_half = right_embeds[:, :n_half, :]  # 우측 이미지 좌측 절반
        
        # 1. 유사성(invariance) 손실 - 겹치는 영역이 유사해야 함
        # 배치 내 각 쌍에 대해 MSE 손실 계산
        sim_loss = F.mse_loss(left_right_half, right_left_half)
        
        # 2. 분산(variance) 손실 - 각 차원이 충분한 분산을 가져야 함
        # 평균 제거
        left_centered = left_right_half - left_right_half.mean(dim=0, keepdim=True)
        right_centered = right_left_half - right_left_half.mean(dim=0, keepdim=True)
        
        # 표준 편차 계산
        std_left = torch.sqrt(left_centered.var(dim=0) + self.eps)
        std_right = torch.sqrt(right_centered.var(dim=0) + self.eps)
        
        # 분산 손실: 각 차원의 표준 편차가 1보다 작으면 페널티
        var_loss = torch.mean(F.relu(1 - std_left)) + torch.mean(F.relu(1 - std_right))
        
        # 3. 공분산(covariance) 손실 - 특징 간 상관관계 최소화
        cov_left = (left_centered.T @ left_centered) / (B - 1)
        cov_right = (right_centered.T @ right_centered) / (B - 1)
        
        # 대각선 마스크 (대각 요소 제외)
        mask = ~torch.eye(D, device=left_embeds.device, dtype=torch.bool)
        
        # 비대각 요소의 제곱합
        cov_loss = (cov_left[mask]**2).mean() + (cov_right[mask]**2).mean()
        
        # 전체 손실 계산
        total_loss = self.sim_coef * sim_loss + self.var_coef * var_loss + self.cov_coef * cov_loss
        
        return total_loss, {
            "sim_loss": sim_loss.detach(), 
            "var_loss": var_loss.detach(), 
            "cov_loss": cov_loss.detach()
        }