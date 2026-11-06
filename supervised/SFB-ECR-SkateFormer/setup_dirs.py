import os
import sys
import json
import shutil
from pathlib import Path

BASE_DIR = Path(r"d:\Thesis\SFB-ECR-SkateFormer")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Subdirectories
(BASE_DIR / "model").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "feeders").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "concepts").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "torchlight" / "torchlight").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "config" / "train" / "ntu_cs").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "config" / "train" / "ntu_cv").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "config" / "test" / "ntu_cs").mkdir(parents=True, exist_ok=True)

# Copy concept CSV from ECR-SkateFormer if exists
src_csv = Path(r"d:\Thesis\ECR-SkateFormer\concepts\ntu_rgbd_60_120.csv")
dst_csv = BASE_DIR / "concepts" / "ntu_rgbd_60_120.csv"
if src_csv.exists():
    shutil.copyfile(src_csv, dst_csv)
    print(f"Copied {src_csv} to {dst_csv}")

print("Directory structure created successfully.")
