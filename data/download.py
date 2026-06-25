"""
download.py - Download WildReceipt and CORD automatically.

SROIE requires a Kaggle account and cannot be fully automated without
API credentials, so it stays a manual step - see the printed instructions
below, or README.md.

Usage: python data/download.py
"""

import os
import tarfile
import urllib.request

WILDRECEIPT_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
WILDRECEIPT_DIR = "data/raw/wildreceipt"
SROIE_DIR = "data/raw/sroie"


def download_wildreceipt():
    os.makedirs(WILDRECEIPT_DIR, exist_ok=True)
    tar_path = os.path.join(WILDRECEIPT_DIR, "wildreceipt.tar")
    extracted_marker = os.path.join(WILDRECEIPT_DIR, "wildreceipt", "train.txt")

    if os.path.exists(extracted_marker):
        print("[WildReceipt] already downloaded and extracted, skipping.")
        return

    if not os.path.exists(tar_path):
        print(f"[WildReceipt] downloading from {WILDRECEIPT_URL} ...")
        urllib.request.urlretrieve(WILDRECEIPT_URL, tar_path)
        print("[WildReceipt] download complete.")

    print("[WildReceipt] extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(WILDRECEIPT_DIR)
    print("[WildReceipt] extraction complete.")


def download_cord():
    from datasets import load_dataset
    print("[CORD] downloading via HuggingFace datasets (cached after first run)...")
    ds = load_dataset("naver-clova-ix/cord-v2")
    print("[CORD] splits:", {k: len(v) for k, v in ds.items()})


def check_sroie():
    marker = os.path.join(SROIE_DIR, "SROIE2019", "train", "box")
    if os.path.isdir(marker):
        print("[SROIE] found, skipping manual download instructions.")
        return

    print("\n" + "=" * 60)
    print("[SROIE] NOT FOUND - manual download required")
    print("=" * 60)
    print("""
SROIE requires a free Kaggle account and cannot be auto-downloaded here.

Steps:
  1. Create a free account at https://www.kaggle.com (if you don't have one)
  2. Go to: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
  3. Click "Download" -> "Download dataset as zip" (~875 MB)
  4. Move the downloaded zip into this project, then run:

       mkdir -p data/raw/sroie
       mv ~/Downloads/archive.zip data/raw/sroie/   # adjust filename/path as needed
       cd data/raw/sroie && unzip archive.zip && cd -

  5. Re-run this script (python data/download.py) to confirm SROIE is found.
""")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Dataset Download")
    print("=" * 60)
    download_wildreceipt()
    download_cord()
    check_sroie()
    print("\nDone. Run 'python data/preprocess.py' next once all 3 sources are present.")