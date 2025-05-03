import torch
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision

# 데이터 로드
ds = load_dataset(
    "csv",
    data_files={
        "train": "data/raw/QuIC360/train.csv",
        "valid": "data/raw/QuIC360/valid.csv",
        "test": "data/raw/QuIC360/test.csv"
    }
)       # ds["train","valid","test"][0] : {"url": "https://...jpg", "query": "...", "annotation", "..."}

PATCH_SIZE = 224
STRIDE     = PATCH_SIZE // 2
to_tensor  = T.ToTensor()

def transform_batch(batch):
    all_batch_pixel_values = []  # 배치의 모든 이미지에 대한 패치들을 저장
    
    for url in batch["url"]:
        image_pixel_values = []  # 현재 이미지의 패치들을 저장
        
        # 1) 이미지 다운로드
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")

        # 2) 리사이즈 (height 고정)
        w, h = img.size
        new_h = PATCH_SIZE
        new_w = int(w * (new_h / h))
        print("리사이즈 전:", img.size, "-> 리사이즈 후:", (new_w, new_h))
        img_r = img.resize((new_w, new_h), Image.BILINEAR)

        # 3) 슬라이딩 크롭
        for x in range(0, new_w - PATCH_SIZE + 1, STRIDE):
            patch = img_r.crop((x, 0, x + PATCH_SIZE, PATCH_SIZE))
            image_pixel_values.append(to_tensor(patch))

        # 4) 래핑 패치 (끝과 처음 50%)
        last = img_r.crop((new_w - STRIDE, 0, new_w, PATCH_SIZE))
        first = img_r.crop((0, 0, STRIDE, PATCH_SIZE))
        wrap = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
        wrap.paste(last, (0, 0))
        wrap.paste(first, (STRIDE, 0))
        image_pixel_values.append(to_tensor(wrap))
        
        # 현재 이미지의 모든 패치를 스택하여 저장
        all_batch_pixel_values.append(torch.stack(image_pixel_values))
    
    # 배치의 모든 이미지 패치를 스택
    batch["pixel_values"] = torch.stack(all_batch_pixel_values)
    
    return batch

# 실시간 transform 설정
ds = ds.with_transform(transform_batch)


from py360convert import e2p

# arrays: List[np.ndarray], 각각 shape (H, W, C), dtype=uint8 또는 float
def rectify_batch(arrays, fov_deg=90, theta=0, phi=0, height=512, width=512):
    rectified = []
    for arr in arrays:
        # e2p는 (H, W, C) uint8 또는 float 입력을 바로 받습니다.
        persp = e2p(
            arr,
            fov_deg=fov_deg,
            u_divs=width,
            v_divs=height,
            theta_deg=theta,
            phi_deg=phi,
            roll_deg=0,
        )
        rectified.append(persp)
    return rectified

# 사용 예
rectified_images = rectify_batch(
    arrays=your_numpy_images,
    fov_deg=90,
    theta=0,   # 필요에 따라 조정
    phi=0,
    height=512,
    width=512
)

#Test Code
if __name__ == "__main__":
    # DataLoader (batch_size=1)
    loader = DataLoader(ds["train"], batch_size=1)

    # 한 배치 시각화
    batch = next(iter(loader))
    patches = batch["pixel_values"][0]  # 첫 번째 배치의 모든 패치, shape: [패치수, 3, 224, 224]
    print("pixel_values 전체 shape:", batch["pixel_values"].shape)
    print("첫 번째 배치의 patches shape:", patches.shape)
    num_patches = patches.size(0)

    plt.figure(figsize=(4 * num_patches, 4))
    for i in range(num_patches):
        ax = plt.subplot(1, num_patches, i + 1)
        
        img = patches[i].permute(1, 2, 0).cpu().numpy()  # (224,224,3)
        ax.imshow(img)
        ax.set_title(f"Patch {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
