# Kaggle Gateway

Thin HTTP gateway for **Chat / MCP → Vercel → Kaggle**.

The gateway keeps Kaggle credentials on the server and exposes only a narrow set of operations for triggering an existing Kaggle kernel, polling its status, and reading kernel / dataset outputs.

## Vercel environment variables

Configure these in **Vercel → Project → Settings → Environment Variables**.

### Required

- `KAGGLE_API_TOKEN` — Kaggle API token. Keep this secret; never commit it to GitHub.

### Recommended

- `GATEWAY_API_KEY` — a private bearer token protecting the gateway endpoints. Any long random secret is fine. When configured, requests must send:
  `Authorization: Bearer <GATEWAY_API_KEY>`

## API

- `GET /api/health` — check Vercel → Kaggle authentication.
- `POST /api/h3/run` — create a new version of an existing Kaggle kernel and trigger execution.
- `GET /api/h3/status?kernel=owner/kernel-slug` — poll kernel execution status.
- `GET /api/h3/output-files?kernel=owner/kernel-slug` — list kernel output files.
- `GET /api/h3/output?kernel=owner/kernel-slug&file=...` — redirect to a selected kernel output file.
- `GET /api/dataset/files?dataset=owner/dataset-slug` — list dataset files.
- `GET /api/dataset/download?dataset=owner/dataset-slug&file=...` — redirect to a selected dataset file.

Interactive OpenAPI docs are available at `/docs`.

## H3 execution model

`POST /api/h3/run` does **not** wait for the Kaggle notebook to finish. It pulls the current kernel source and metadata into Vercel's temporary workspace, pushes a new version to the same Kaggle kernel, and returns after Kaggle accepts the new version. Use `/api/h3/status` to poll execution state and the output endpoints after completion.

## Security

- Kaggle credentials are read only from Vercel environment variables.
- Secrets are never returned to the browser or intentionally written to logs.
- Kernel / dataset identifiers are validated.
- Arbitrary shell commands are not exposed.
- Temporary kernel source is stored only under `/tmp` during the request.

## Deploy

Import this GitHub repository into Vercel, keep the repository root as the project root, add the environment variables above, and deploy. Vercel should detect the Python / FastAPI application automatically.
