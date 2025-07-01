import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VICRegLoss(nn.Module):
    """
    VICReg 손실 함수의 개선된 구현.
    효율성, 안정성, 가독성에 중점을 두어 리팩토링되었습니다.
    """
    def __init__(self, sim_coef=25.0, var_coef=25.0, cov_coef=1.0, eps=1e-5):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x (torch.Tensor): 첫 번째 임베딩. 형태: (B, N, D) 또는 (B, H, W, D)
            y (torch.Tensor): 두 번째 임베딩. 형태: (B, N, D) 또는 (B, H, W, D)
        
        Returns:
            Tuple[torch.Tensor, dict]: 총 손실과 개별 손실 컴포넌트.
        """
        # 1. 입력 전처리: (B, ..., D) 형태를 (B, N, D) 형태로 평탄화
        original_shape = x.shape
        x = x.reshape(original_shape[0], -1, original_shape[-1])
        y = y.reshape(original_shape[0], -1, original_shape[-1])
        
        B, N, D = x.shape

        # 2. Invariance Loss (유사성 손실)
        sim_loss = F.mse_loss(x, y)

        # 3. Variance Loss (분산 손실)
        # 각 임베딩의 배치 내 표준편차를 1에 가깝게 유지
        std_x = torch.sqrt(x.var(dim=1) + self.eps)
        std_y = torch.sqrt(y.var(dim=1) + self.eps)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # 4. Covariance Loss (공분산 손실)
        # 각 임베딩의 특징(feature) 차원 간 상관관계를 0으로 만듦
        # 평균을 0으로 만들어 중앙화
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        
        # 공분산 행렬 계산 (효율적인 배치 행렬 곱셈 활용)
        # (B, N, D) -> (B, D, N)
        x_centered = x_centered.transpose(1, 2)
        y_centered = y_centered.transpose(1, 2)
        
        # cov_x: (B, D, N) @ (B, N, D) -> (B, D, D)
        cov_x = (x_centered @ x_centered.transpose(1, 2)) / (N - 1)
        cov_y = (y_centered @ y_centered.transpose(1, 2)) / (N - 1)

        # 대각선 요소를 제외한 나머지(off-diagonal) 값들의 제곱합을 최소화
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=x.device)
        cov_loss = (cov_x[:, off_diag_mask].pow(2).sum() / D + 
                    cov_y[:, off_diag_mask].pow(2).sum() / D) / B

        # 5. 최종 손실 계산
        total_loss = (
            self.sim_coef * sim_loss +
            self.var_coef * var_loss +
            self.cov_coef * cov_loss
        )
        
        losses_log = {
            "sim_loss": sim_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach()
        }
        return total_loss, losses_log