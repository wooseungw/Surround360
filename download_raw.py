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

async def download_image(session, url, cache_path, semaphore, rate_limiter, max_retries=3):
    """비동기 이미지 다운로드. 성공 시 로컬 경로, 실패 시 None 반환."""
    async with semaphore:
        if os.path.exists(cache_path):
            return cache_path
        
        await rate_limiter.wait()
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get('Retry-After', 60))
                        print(f"[RATE_LIMIT] {url} - 대기: {retry_after}초")
                        await asyncio.sleep(retry_after)
                        continue  # 재시도
                    
                    if resp.status == 200:
                        # Content-Type 확인
                        content_type = resp.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                            print(f"[WARNING] {url} - 이미지가 아닌 콘텐츠: {content_type}")
                            return None
                        
                        data = await resp.read()
                        if len(data) == 0:
                            print(f"[WARNING] {url} - 빈 파일")
                            return None
                            
                        # 디렉토리 생성
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        
                        with open(cache_path, 'wb') as f:
                            f.write(data)
                        return cache_path
                    else:
                        print(f"[HTTP_ERROR] {url} - Status: {resp.status}")
                        if attempt == max_retries - 1:  # 마지막 시도
                            return None
                        await asyncio.sleep(2 ** attempt)  # 지수적 백오프
                        
            except asyncio.TimeoutError:
                print(f"[TIMEOUT] {url} - 시도 {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                print(f"[ERROR] {url} - 시도 {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
                
        return None

async def batch_download_unique(urls, cache_dir, max_concurrent, rps):
    """
    중복 URL 제거 후 한 번씩만 다운로드,
    반환값: { url: local_path or None, ... }
    """
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(max_concurrent)
    limiter = RateLimiter(rps)
    
    # SSL 및 타임아웃 설정
    connector = aiohttp.TCPConnector(
        ssl=False,  # SSL 검증 비활성화 (필요시)
        limit=100,
        limit_per_host=max_concurrent
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        headers=headers
    ) as session:
        tasks = []
        for url in urls:
            if not url or pd.isna(url):  # URL 유효성 검사
                continue
                
            fname = os.path.basename(url.split('?')[0])  # 쿼리 파라미터 제거
            if not fname or '.' not in fname:
                fname = f"image_{hash(url) % 1000000}.jpg"  # 기본 파일명
                
            path = os.path.join(cache_dir, fname)
            tasks.append(download_image(session, url, path, sem, limiter))
            
        if not tasks:
            print("[WARNING] 다운로드할 유효한 URL이 없습니다.")
            return {}
            
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
    print(f"📖 CSV 파일 로드 중: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   총 {len(df)} 행 발견")
    
    # URL 컬럼 존재 확인
    if url_col not in df.columns:
        print(f"❌ ERROR: '{url_col}' 컬럼이 CSV에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        return
    
    # 유효한 URL 필터링
    valid_urls = df[url_col].dropna()
    print(f"   유효한 URL: {len(valid_urls)} 개")

    # 2) 유니크 URL 다운로드
    unique_urls = valid_urls.unique().tolist()
    print(f"   중복 제거 후: {len(unique_urls)} 개 다운로드 예정")
    
    if not unique_urls:
        print("⚠️ WARNING: 다운로드할 URL이 없습니다.")
        return
    
    url2path = asyncio.run(batch_download_unique(
        unique_urls,
        cache_dir=image_dir,
        max_concurrent=max_concurrent,
        rps=rps
    ))

    # 3) 매핑 & 필터링
    df['local_path'] = df[url_col].map(url2path)
    success_count = df['local_path'].notna().sum()
    print(f"✅ 성공적으로 다운로드: {success_count}/{len(df)} 개")
    
    df = df[df['local_path'].notna()].copy()
    
    if len(df) == 0:
        print("❌ ERROR: 다운로드된 이미지가 없습니다.")
        return

    # 4) URL 컬럼을 상대경로로 포맷
    def format_path(p: str):
        fname = os.path.basename(p)
        # data/quic360/train/images/img.jpg 와 같이 고정된 상대경로 반환
        return os.path.join(out_dir, split, "images", fname)

    df[url_col] = df['local_path'].apply(format_path)
    df.drop(columns=['local_path'], inplace=True)

    # 5) 결과 저장
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ [{Path(input_csv).name}] → {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="data/raw/QuIC360/*.csv → 이미지 다운로드 + CSV 갱신"
    )
    parser.add_argument("--splits", nargs="*",
                        help="처리할 split 이름(예: train test). 생략 시 raw_dir/*.csv 전부")
    parser.add_argument("--raw_dir", default="data/raw/QuIC360",
                        help="원본 CSV 디렉토리 (default: data/raw/QuIC360)")
    parser.add_argument("--out_dir", default="data/quic360",
                        help="출력 베이스 디렉토리 (default: data/quic360)")
    parser.add_argument("--url_col", default="url", help="URL 컬럼명")
    parser.add_argument("--max_concurrent", type=int, default=8, help="동시 다운로드 수")
    parser.add_argument("--requests_per_second", type=float, default=0.7, help="초당 요청 수 제한")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = args.out_dir  # e.g. "data/quic360"

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