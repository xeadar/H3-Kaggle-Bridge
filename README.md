# H3 Kaggle Bridge

Control plane for **ChatGPT → GitHub → GitHub Actions → Kaggle**.

The repository is intentionally not a web server. GitHub Actions is the execution layer; Kaggle remains the compute/data platform.

## Current bridge test

The first implemented path is deliberately non-GPU and does not touch the active Garden Tales notebook run:

1. Read `control/first-frame-request.json`.
2. Authenticate to Kaggle using the GitHub Actions secret `KAGGLE_API_TOKEN`.
3. Read the existing first-frame asset from `sita2ksitas/h3-first-frame-test`.
4. Copy it into the separate private Dataset `sita2ksitas/h3-kaggle-bridge-inputs`.
5. Verify the uploaded file is visible from Kaggle.
6. Write the result back to `results/first-frame-upload.json`.

Successful completion proves the path:

`ChatGPT → GitHub → Actions → Kaggle API/CLI → Kaggle Dataset → GitHub result`

## Required GitHub Actions secret

Repository **Settings → Secrets and variables → Actions → New repository secret**

- Name: `KAGGLE_API_TOKEN`
- Value: the Kaggle API token from the Kaggle account settings.

Never commit the token to the repository.

## Files

- `.github/workflows/first-frame-to-kaggle.yml` — first-frame Dataset transfer and verification.
- `control/first-frame-request.json` — request/control payload.
- `results/first-frame-upload.json` — written by Actions after a run.

## Safety boundary for this test

This workflow does **not** start a Kaggle Notebook, request a GPU, modify Garden Tales v19, or write to the existing `h3-first-frame-test` Dataset. The source Dataset is read-only during the test; the destination is a separate private bridge Dataset.
