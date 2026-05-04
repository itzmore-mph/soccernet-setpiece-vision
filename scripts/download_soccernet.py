"""
Download SoccerNet GSR Dataset to external SSD.
Idempotent: splits already on disk are skipped.
Run: python scripts/download_soccernet.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader

load_dotenv()

password = os.getenv("SOCCERNET_PASSWORD")
if not password:
    raise EnvironmentError("SOCCERNET_PASSWORD not set in .env")

TASK = "gamestate-2024"
ALL_SPLITS = ["train", "valid", "test", "challenge"]

LOCAL_DIR = Path(os.getenv(
    "SOCCERNET_LOCAL_DIR",
    "/Volumes/MPH-ExternalStorage/soccernet-gsr",
))
TASK_DIR = LOCAL_DIR / TASK


def split_present(split: str) -> bool:
    """Split counts as downloaded if zip exists (>0 bytes) or extracted dir non-empty."""
    zip_path = TASK_DIR / f"{split}.zip"
    extracted = TASK_DIR / split
    if zip_path.is_file() and zip_path.stat().st_size > 0:
        return True
    if extracted.is_dir() and any(extracted.iterdir()):
        return True
    return False


print(f"Target: {LOCAL_DIR}")

todo = [s for s in ALL_SPLITS if not split_present(s)]
done = [s for s in ALL_SPLITS if split_present(s)]

for s in done:
    print(f"  skip {s} (already present)")

if not todo:
    print("All splits present. Nothing to download.")
else:
    print(f"Downloading splits: {todo}")
    dl = SoccerNetDownloader(LocalDirectory=str(LOCAL_DIR))
    dl.password = password  # type: ignore[assignment]
    dl.downloadDataTask(task=TASK, split=todo)
    print(f"Done. Data saved to {LOCAL_DIR}")
