# TVCalib Environment (homography regeneration only)

TVCalib is a third-party tool (MM4SPA/tvcalib) used once to generate the
committed homographies in outputs/homographies_tvcalib.parquet. It is NOT
part of the standard reproduction flow. This folder documents the exact
environment for optional regeneration from raw video.

## Setup
1. Clone alongside the main repo: `git clone https://github.com/MM4SPA/tvcalib`
2. Create env: `cd tvcalib && python3.11 -m venv .venv && source .venv/bin/activate`
3. Install pinned deps: `pip install -r ../soccernet-setpiece-vision/docs/tvcalib-setup/tvcalib-requirements-frozen.txt`
4. Apply two PyTorch 2.x patches (see below)
5. Place segmentation checkpoint at tvcalib/data/segment_localization/train_59.pt

## Patches
- tvcalib/tvcalib/sncalib_dataset.py line 13: replace `from torch._six import string_classes` with `string_classes = (str, bytes)`
- tvcalib/run_inference.py line 89: replace `torch.load(path)` with `torch.load(path, weights_only=False)`
