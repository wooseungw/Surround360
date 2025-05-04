import asyncio
import aiohttp
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm

class RateLimiter:
    """
    초당 요청 수(requests_per_second)를 제한하는 간단한 레이트 리미터
    """
    def __init__(self, requests_per_second: float):
        self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_call = None

    async def wait(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._last_call is not None:
                elapsed = now - self._last_call
                wait_time = self._interval - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            self._last_call = asyncio.get_event_loop().time()

async def download_image(session, url, cache_path, semaphore, rate_limiter):
    """비동기 이미지 다운로드. 성공 시 로컬 경로, 실패 시 None 반환."""
    async with semaphore:
        if os.path.exists(cache_path):
            return cache_path
        await rate_limiter.wait()
        try:
            async with session.get(url) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return await download_image(session, url, cache_path, semaphore, rate_limiter)
                if resp.status == 200:
                    data = await resp.read()
                    with open(cache_path, 'wb') as f:
                        f.write(data)
                    return cache_path
        except Exception as e:
            print(f"[ERROR] {url} 다운로드 실패: {e}")
        return None

async def batch_download_unique(urls, cache_dir, max_concurrent, rps):
    """
    중복 URL 제거 후 한 번씩만 다운로드,
    반환값: { url: local_path or None, ... }
    """
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)
    limiter = RateLimiter(rps)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            fname = os.path.basename(url)
            path = os.path.join(cache_dir, fname)
            tasks.append(download_image(session, url, path, sem, limiter))
        results = await tqdm.gather(*tasks, desc="Downloading unique images")
    return dict(zip(urls, results))

def process_split(
    input_csv: str,
    output_csv: str,
    image_dir: str,
    url_col: str,
    max_concurrent: int,
    rps: float,
    out_dir: str,
    split: str
):
    # 1) CSV 로드
    df = pd.read_csv(input_csv)

    # 2) 유니크 URL 다운로드
    unique_urls = df[url_col].unique().tolist()
    url2path = asyncio.run(batch_download_unique(
        unique_urls,
        cache_dir=image_dir,
        max_concurrent=max_concurrent,
        rps=rps
    ))

    # 3) 매핑 & 필터링
    df['local_path'] = df[url_col].map(url2path)
    df = df[df['local_path'].notna()].copy()

    # 4) URL 컬럼을 "data/{split}/images/{filename}" 로 고정
    def format_path(p: str):
        fname = os.path.basename(p)
        # 항상 data/… 로 시작하도록 하려면 out_dir 인자가 "data" 여야 합니다.
        cur_dir = os.getcwd().split("\\")[-1]
        return f"{cur_dir}/{out_dir}/{split}/images/{fname}"

    df[url_col] = df['local_path'].apply(format_path)
    df.drop(columns=['local_path'], inplace=True)

    # 5) 결과 저장
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ [{Path(input_csv).name}] → {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="raw_dir/*.csv → 이미지 다운로드 + CSV 갱신"
    )
    parser.add_argument("--splits", nargs="*",
                        help="처리할 split(예: train test). 생략 시 raw_dir/*.csv 전부")
    parser.add_argument("--raw_dir", default="raw/QuIC360",
                        help="원본 CSV 디렉토리")
    parser.add_argument("--out_dir", default="quic360",
                        help="출력 베이스 디렉토리 (맨 앞에 붙일 폴더명)")
    parser.add_argument("--url_col", default="url", help="URL 컬럼명")
    parser.add_argument("--max_concurrent", type=int, default=8, help="동시 다운로드 수")
    parser.add_argument("--requests_per_second", type=float, default=0.7, help="초당 요청 수 제한")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = args.out_dir  # 여기서는 문자열 "quic360" 가 되길 기대

    if args.splits:
        splits = args.splits
    else:
        splits = [p.stem for p in raw_dir.glob("*.csv")]

    for split in splits:
        print(f"🔄 [{split}] 처리 중…")
        in_csv  = raw_dir / f"{split}.csv"
        out_csv = Path(out_dir) / f"{split}.csv"
        img_dir = Path(out_dir) / split / "images"

        if not in_csv.exists():
            print(f"⚠️ 파일 없음: {in_csv}, 건너뜁니다.")
            continue

        process_split(
            input_csv=str(in_csv),
            output_csv=str(out_csv),
            image_dir=str(img_dir),
            url_col=args.url_col,
            max_concurrent=args.max_concurrent,
            rps=args.requests_per_second,
            out_dir=out_dir,
            split=split
        )