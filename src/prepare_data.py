"""data/raw/클래스명/의 이미지를 train/val/test로 분리합니다."""
import argparse
import random
import shutil
from pathlib import Path

from src.config import PROCESSED_DIR, RAW_DIR, SEED

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def split_counts(n: int) -> tuple[int, int, int]:
    """소수 데이터에서도 각 집합에 이미지가 1장 이상 있도록 분할합니다."""
    if n < 3:
        raise ValueError("각 클래스에 이미지가 최소 3장 필요합니다. 실제 학습에는 훨씬 더 많이 권장합니다.")
    train_count = max(1, int(n * 0.70))
    val_count = max(1, int(n * 0.15))
    if train_count + val_count >= n:
        train_count = n - val_count - 1
    return train_count, val_count, n - train_count - val_count


def main(overwrite: bool = False) -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"원본 데이터 폴더를 찾을 수 없습니다: {RAW_DIR}")

    class_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("data/raw/ 아래에 서로 다른 클래스 폴더가 최소 2개 필요합니다.")

    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{PROCESSED_DIR}에 기존 분할 결과가 있습니다. 새로 분할하려면 --overwrite를 사용하세요."
            )

    # 사전에 검증하고 나서 기존 결과를 지웁니다.
    class_files: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        split_counts(len(files))
        class_files[class_dir.name] = files

    if PROCESSED_DIR.exists() and overwrite:
        shutil.rmtree(PROCESSED_DIR)

    rng = random.Random(SEED)
    for class_name, files in class_files.items():
        rng.shuffle(files)
        n_train, n_val, _ = split_counts(len(files))
        partitions = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, images in partitions.items():
            target_dir = PROCESSED_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in images:
                shutil.copy2(source, target_dir / source.name)
        print(f"{class_name}: " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))

    print(f"분할 완료: {PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습/검증/테스트 이미지 분리")
    parser.add_argument("--overwrite", action="store_true", help="기존 processed/ 폴더를 삭제하고 재생성")
    args = parser.parse_args()
    main(overwrite=args.overwrite)