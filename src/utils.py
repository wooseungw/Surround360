import torch
import matplotlib.pyplot as plt

def visualize_tensor_batch(tensor_batch):
    """
    Visualize a batch of images stored as a tensor of shape (B, 3, H, W).
    Each image is displayed in a grid layout.
    """
    B, C, H, W = tensor_batch.shape
    
    # 격자 형태의 행/열 개수 계산 (정사각형에 가깝게)
    rows = 1
    cols = B
    

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for idx in range(B):
        # (3, H, W) → (H, W, 3) 형태로 변환
        img = tensor_batch[idx].permute(1, 2, 0).cpu().numpy()
        axes[idx].imshow(img)
        axes[idx].axis('off')

    # 사용하지 않은 서브플롯은 꺼두기
    for idx in range(B, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

