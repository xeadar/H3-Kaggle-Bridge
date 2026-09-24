# H3 Kaggle Bridge

Control plane for **ChatGPT → GitHub → GitHub Actions → Kaggle**.

GitHub Actions is the execution layer; Kaggle remains the compute/data platform.

## Current bridge test

The active test is deliberately read-only against Kaggle and does not touch the running Garden Tales notebook or GPU session.

1. Read `control/fetch-baseline-video.json`.
2. Authenticate to Kaggle using the GitHub Actions secret `KAGGLE_API_TOKEN`.
3. Download the private Dataset `sita2ksitas/garden-tales-pv`.
4. Locate the known Version 16 baseline MP4 by its SHA256, not by a guessed filename.
5. Verify its byte size and SHA256.
6. Upload the MP4 plus a receipt as the GitHub Actions artifact `garden-tales-v16-baseline`.
7. ChatGPT can inspect the workflow result and retrieve the artifact through the connected GitHub bridge.

Successful completion proves the return path:

`ChatGPT → GitHub → Actions → Kaggle Dataset → Actions Artifact → GitHub → ChatGPT`

## Required GitHub Actions secret

Repository **Settings → Secrets and variables → Actions → New repository secret**

- Name: `KAGGLE_API_TOKEN`
- Value: the Kaggle API token from the Kaggle account settings.

Never commit the token to the repository or paste it into a control file.

## Files

- `.github/workflows/fetch-baseline-video.yml` — read-only Kaggle Dataset fetch and artifact return.
- `control/fetch-baseline-video.json` — request payload; creating or changing it triggers the workflow.
- `results/fetch-baseline-video.json` — workflow result written back by Actions.

## Safety boundary

This workflow does **not** start a Kaggle Notebook, request a GPU, create a Dataset version, modify Garden Tales v19, or alter `Garden Tales PV`. It only downloads existing Dataset content and returns one already-known MP4 through GitHub Actions.
