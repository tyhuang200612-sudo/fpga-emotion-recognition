# FER2013 public dataset: Kaggle + GitHub Actions

## What is already implemented

- Three selected folder names: `neutral`, `happy`, `sad`.
- FPGA label IDs: `neutral=0`, `happy=1`, `sad=2`. These IDs are **project IDs**, not the original seven-class dataset's numeric indexes.
- Input comes from Kaggle public dataset `msambare/fer2013`; no 3.94GB private ZIP upload required for this baseline.
- Kaggle audit script: `tools/prepare_fer2013.py`.
- GitHub Actions manual job: `.github/workflows/kaggle-audit.yml`.
- The audit job **does not train** and `enable_gpu=false`. It creates `/kaggle/working/fer2013-audit/{train,val,test}.csv` and `audit.json`.
- Keep Kaggle's original held-out `test` partition. Generate a validation set from original `train` without mixing exact duplicate hashes.
- Person/video identity information is absent in this FER2013 variant: grouping by exact file hashes does **not** prove there is no person/near-duplicate leakage.
- The original guide's proposed 70/15/15 is not exact here, because FER2013's published train/test partition is retained; record the actual counts instead.

## One-time setup (no training yet)

1. Check the [FER2013 Kaggle dataset](https://www.kaggle.com/datasets/msambare/fer2013). It contains a train/test image-folder dataset with seven facial-expression classes. Review its usage terms and any notices.
2. In your private GitHub repository, visit **Settings → Secrets and variables → Actions → New repository secret**.
3. Add a secret named `KAGGLE_API_TOKEN`. Generate your own API token from [Kaggle API settings](https://www.kaggle.com/settings/api). Do not commit or paste the token into notebooks or chat.
4. Visit [GitHub Actions](https://github.com/tyhuang200612-sudo/fpga-emotion-recognition/actions), select **Kaggle FER2013 audit (CPU only, manual)** → **Run workflow**.
5. Enter your **Kaggle username** (not email) and start. GitHub Actions packages and submits a **private, CPU-only Kaggle script**, attaching `msambare/fer2013` as a dataset source. No GPU credits are spent by this workflow.
6. See `https://www.kaggle.com/code/<your-username>/fpga-emotion-audit` for Kaggle job status and **Output** files. The GitHub Action confirms submission; it does **not** prove the remote audit has finished.
7. Review `audit.json`: per-class counts, unreadable pictures, conflicting labels, cross-split exact hashes, and dataset caveats. Confirm that the class definitions and license are appropriate before training.

If you don't want to set up an API token yet, click **New Notebook** on the public Kaggle dataset page, make sure the dataset is listed in Notebook **Input**, and try this CPU-only count:

```python
from pathlib import Path
root = Path("/kaggle/input/fer2013")
for split in ["train", "test"]:
    for emotion in ["neutral", "happy", "sad"]:
        folder = root / split / emotion
        print(split, emotion, len(list(folder.glob("*"))) if folder.exists() else "folder not found")
```

The exact input mount directory may vary. Check the Kaggle Notebook **Input** panel if the folder is not found.

## More automation later

- After approval of real dataset audit, implement FP32 Tiny-CNN train/validate/test, INT8 PTQ/QAT, integer reference and FPGA weight export.
- Training will require a **different** Kaggle kernel with `enable_gpu=true`, explicit quota checks and result checkpointing.
- The private GitHub repository does not contain raw facial images or downloaded model weights.
- Never claim actual accuracy or RTL resource usage before measured results.
