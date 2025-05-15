import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """메모리 효율적인 VICReg 손실 함수 구현.
    
    원래 VICReg에서 공분산 계산 부분을 최적화하고, 
    선택적 샘플링을 통해 메모리 사용량을 크게 줄였습니다.
    """
    
    def __init__(self, 
                 sim_coef=25.0, 
                 var_coef=25.0, 
                 cov_coef=1.0, 
                 eps=1e-4,
                 max_proj_dim=64):  # 투영 차원 제한
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.eps = eps
        self.max_proj_dim = max_proj_dim
    
    def random_projection(self, x, output_dim):
        """대규모 임베딩을 더 작은 차원으로 무작위 투영"""
        input_dim = x.shape[-1]
        if input_dim <= output_dim:
            return x
            
        # 무작위 투영 행렬 생성 (입력마다 고정된 행렬을 사용하기 위해 seed 설정)
        with torch.no_grad():
            torch.manual_seed(42)  # 재현성을 위한 고정 시드
            proj = torch.randn(input_dim, output_dim, device=x.device, dtype=x.dtype)
            proj = proj / torch.sqrt(torch.sum(proj**2, dim=0, keepdim=True))
            
        # 투영 적용
        return x @ proj
    
    def forward(self, left_embeds, right_embeds, sample_ratio=1.0):
        """
        효율적인 VICReg 손실 계산
        
        Args:
            left_embeds: 좌측 이미지 특징 (B, N, D)
            right_embeds: 우측 이미지 특징 (B, N, D)
            sample_ratio: 계산에 사용할 특징의 비율 (0-1)
            
        Returns:
            total_loss: 전체 손실
            losses: 개별 손실 컴포넌트 (dictionary)
        """
        # 1. 선택적 샘플링으로 계산량 감소
        if sample_ratio < 1.0:
            N = left_embeds.size(1)
            sample_size = max(1, int(N * sample_ratio))
            indices = torch.randperm(N, device=left_embeds.device)[:sample_size]
            left_embeds = left_embeds[:, indices]
            right_embeds = right_embeds[:, indices]
        
        # 원본 형태 유지
        B, N, D = left_embeds.shape
        
        # 2. 차원 감소로 메모리 사용량 감소
        if D > self.max_proj_dim:
            left_proj = self.random_projection(left_embeds, self.max_proj_dim)
            right_proj = self.random_projection(right_embeds, self.max_proj_dim)
        else:
            left_proj = left_embeds
            right_proj = right_embeds
            
        # 3. 효율적인 유사성 손실 계산 (변경 없음)
        sim_loss = F.mse_loss(left_embeds, right_embeds)
        
        # 4. 효율적인 분산 손실 계산
        # 평균 제거
        left_centered = left_embeds - left_embeds.mean(dim=1, keepdim=True)
        right_centered = right_embeds - right_embeds.mean(dim=1, keepdim=True)
        
        # 표준 편차 계산 (메모리 효율적 방식)
        std_left = torch.sqrt(torch.var(left_centered, dim=1) + self.eps)
        std_right = torch.sqrt(torch.var(right_centered, dim=1) + self.eps)
        
        var_loss = (torch.mean(F.relu(1 - std_left)) + 
                   torch.mean(F.relu(1 - std_right))) / 2.0
        
        # 5. 효율적인 공분산 손실 계산
        # 차원 감소된 데이터만 사용하여 메모리 사용량 감소
        B, N, D_proj = left_proj.shape
        
        # 효율적인 공분산 계산을 위해 배치 처리
        cov_loss = 0.0
        batch_size = min(B, 16)  # 배치 크기 제한
        
        for b_idx in range(0, B, batch_size):
            b_end = min(b_idx + batch_size, B)
            curr_batch_size = b_end - b_idx
            
            # 현재 미니배치에 대한 데이터 준비
            left_batch = left_proj[b_idx:b_end].reshape(curr_batch_size, -1)  # (batch, N*D_proj)
            right_batch = right_proj[b_idx:b_end].reshape(curr_batch_size, -1)
            
            # 정규화
            left_batch = left_batch - left_batch.mean(dim=0, keepdim=True)
            right_batch = right_batch - right_batch.mean(dim=0, keepdim=True)
            
            # 경제적인 방식으로 대각 요소를 제외한 공분산 계산
            if curr_batch_size > 1:
                # 공분산 행렬 크기를 제한하기 위해 차원 감소 적용
                max_cov_dim = min(left_batch.size(1), 1024)  # 공분산 계산에 사용할 최대 차원
                if left_batch.size(1) > max_cov_dim:
                    idx = torch.randperm(left_batch.size(1))[:max_cov_dim]
                    left_batch = left_batch[:, idx]
                    right_batch = right_batch[:, idx]
                
                # 효율적인 공분산 계산
                cov_left = (left_batch.T @ left_batch) / (curr_batch_size - 1)
                cov_right = (right_batch.T @ right_batch) / (curr_batch_size - 1)
                
                # 대각선 마스크
                diag = torch.eye(cov_left.size(0), device=cov_left.device, dtype=cov_left.dtype)
                
                # 비대각 요소만 사용
                batch_cov_loss = ((cov_left * (1 - diag)).pow(2).sum() + 
                                 (cov_right * (1 - diag)).pow(2).sum()) / 2
                
                cov_loss += batch_cov_loss
            
        # 배치 수로 정규화
        cov_loss = cov_loss / max(1, (B // batch_size))
        
        # 6. 전체 손실 계산
        total_loss = (self.sim_coef * sim_loss + 
                      self.var_coef * var_loss + 
                      self.cov_coef * cov_loss)
        
        return total_loss, {
            "sim_loss": sim_loss.detach(), 
            "var_loss": var_loss.detach(), 
            "cov_loss": cov_loss.detach()
        }