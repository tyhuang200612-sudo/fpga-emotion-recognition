# GitHub Actions + Colab: supported workflow

## What this repository can automate

- A GitHub Actions **CPU-based** validation job tests the notebook format and Tiny-CNN forward pass.
- The private repo stores source code and the Colab notebook.
- Training can be performed *interactively* from Colab with a T4 when available, then logs/checkpoints can be explicitly exported (avoid committing sensitive face images).

## What it cannot do with a free Colab account

An Actions job cannot obtain or take over your personal free Colab T4 runtime via GitHub's notebook integration. Running an SSH server, remote runner, or remote desktop in the free Colab runtime is not a supported workaround.

Google Colab FAQ: https://research.google.com/colaboratory/faq.html

## Fully unattended GPU training through GitHub Actions

Use a **dedicated, authorized GPU runner** that GitHub Actions can reach (e.g. a self-hosted GPU computer/VM), then put a training workflow on that runner. This would be separate from Google's free Colab runtime. Paid GitHub GPU larger runners have plan/availability/billing constraints.

GitHub runners: https://docs.github.com/en/actions/concepts/runners/github-hosted-runners

## Current status

- Training **not started**; data not provided or audited.
- CPU tests are **not accuracy measurements**.
- FPGA resource usage, performance, and deployment are **not validated**.
- Model target is neutral/happy/sad, 48×48 grayscale, Tiny-CNN-A (8/16/32), INT8 inference on the secondary HX1P35A.
- Do not upload the raw face dataset or private identity records to the repository.
