import os
import re
import tempfile
from pathlib import Path
from typing import Optional

# Kaggle's Python client may try to create a config directory at import time.
# Vercel's deployment filesystem is read-only except /tmp.
os.environ.setdefault("KAGGLE_CONFIG_DIR", "/tmp/.kaggle")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


app = FastAPI(
    title="Kaggle Gateway",
    version="0.1.0",
    description="Thin HTTP gateway for triggering Kaggle kernels and reading Kaggle outputs/datasets.",
)

REF_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/(?:v)?\d+)?$")
ACC_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class RunRequest(BaseModel):
    kernel: str = Field(..., examples=["owner/kernel-slug"])
    accelerator: Optional[str] = Field(default=None, examples=["NvidiaTeslaT4"])
    timeout_seconds: Optional[int] = Field(default=None, ge=30, le=3600)


def _secret(name: str) -> str:
    return os.getenv(name, "").strip()


def _check_gateway(request: Request) -> None:
    expected = _secret("GATEWAY_API_KEY")
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing gateway bearer token")


def _validate_ref(value: str, kind: str) -> str:
    value = (value or "").strip()
    if not REF_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"Invalid {kind} reference")
    return value


def _validate_file_name(value: str) -> str:
    value = (value or "").strip()
    if not value or len(value) > 512 or "\x00" in value:
        raise HTTPException(status_code=400, detail="Invalid file name")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail="Invalid file name")
    return value


def _safe_message(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    for name in ("KAGGLE_API_TOKEN", "GATEWAY_API_KEY"):
        token = _secret(name)
        if token:
            text = text.replace(token, "[REDACTED]")
    return text[:1200]


def _kaggle() -> KaggleApi:
    if not _secret("KAGGLE_API_TOKEN"):
        raise HTTPException(status_code=503, detail="KAGGLE_API_TOKEN is not configured")
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise HTTPException(status_code=502, detail="Kaggle authentication failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kaggle authentication failed: {_safe_message(exc)}") from exc
    return api


def _enum_or_value(value):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return name
    raw = getattr(value, "value", None)
    if raw is not None and not isinstance(raw, (dict, list, tuple, set)):
        return raw
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _first(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _file_row(item):
    created = _first(item, "creation_date", "creationDate")
    if hasattr(created, "isoformat"):
        created = created.isoformat()
    return {
        "name": _first(item, "file_name", "name", default=""),
        "size": _first(item, "total_bytes", "size", "totalBytes"),
        "creation_date": created,
    }


def _next_token(response):
    return _first(response, "next_page_token", "nextPageToken", default="") or ""


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": _safe_message(exc)},
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggle Gateway</title>
<style>
:root{color-scheme:dark;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
body{margin:0;background:#0b0d10;color:#e8edf2}
main{max-width:980px;margin:0 auto;padding:34px 20px 60px}
h1{font-size:26px;margin:0 0 4px}.muted{color:#8b98a5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:22px}
.card{background:#12161b;border:1px solid #27303a;border-radius:14px;padding:16px}
label{display:block;font-size:12px;color:#9ba7b3;margin:12px 0 6px}
input{width:100%;box-sizing:border-box;background:#090c0f;border:1px solid #303a44;border-radius:8px;color:#fff;padding:10px}
button{margin-top:10px;margin-right:7px;background:#e7edf2;color:#111;border:0;border-radius:8px;padding:9px 12px;font-weight:700;cursor:pointer}
button.secondary{background:#27303a;color:#eef4f8}
pre{white-space:pre-wrap;word-break:break-word;background:#080a0d;border:1px solid #27303a;border-radius:12px;padding:14px;min-height:140px}
code{color:#b8e986}
</style>
</head>
<body>
<main>
<h1>Kaggle Gateway</h1>
<div class="muted">Chat / MCP → Vercel → Kaggle</div>
<div class="grid">
<section class="card">
<h3>Connection</h3>
<label>Gateway API Key（若已設定）</label>
<input id="key" type="password" autocomplete="off" placeholder="Bearer secret">
<button onclick="callApi('/api/health')">Health</button>
<a href="/docs" style="color:#b8e986;margin-left:8px">OpenAPI</a>
</section>
<section class="card">
<h3>Kernel / H3</h3>
<label>owner/kernel-slug</label>
<input id="kernel" placeholder="owner/kernel-slug">
<button onclick="kernelStatus()">Status</button>
<button onclick="kernelRun()">Run</button>
<button class="secondary" onclick="kernelFiles()">Output files</button>
</section>
<section class="card">
<h3>Dataset</h3>
<label>owner/dataset-slug</label>
<input id="dataset" placeholder="owner/dataset-slug">
<button onclick="datasetFiles()">List files</button>
</section>
</div>
<h3>Response</h3>
<pre id="out">Ready.</pre>
</main>
<script>
function headers(extra={}) {
  const h={'Content-Type':'application/json',...extra};
  const k=document.getElementById('key').value.trim();
  if(k) h['Authorization']='Bearer '+k;
  return h;
}
async function callApi(url, opts={}) {
  const out=document.getElementById('out');
  out.textContent='Loading...';
  try {
    const r=await fetch(url,{...opts,headers:headers(opts.headers||{})});
    const t=await r.text();
    let body=t; try{body=JSON.stringify(JSON.parse(t),null,2)}catch(e){}
    out.textContent=`HTTP ${r.status}\n${body}`;
  } catch(e) { out.textContent=String(e); }
}
function kernelRef(){return encodeURIComponent(document.getElementById('kernel').value.trim())}
function datasetRef(){return encodeURIComponent(document.getElementById('dataset').value.trim())}
function kernelStatus(){callApi('/api/h3/status?kernel='+kernelRef())}
function kernelFiles(){callApi('/api/h3/output-files?kernel='+kernelRef())}
function datasetFiles(){callApi('/api/dataset/files?dataset='+datasetRef())}
function kernelRun(){
 const kernel=document.getElementById('kernel').value.trim();
 callApi('/api/h3/run',{method:'POST',body:JSON.stringify({kernel})});
}
</script>
</body>
</html>"""


@app.get("/api/health")
def health(request: Request):
    _check_gateway(request)
    api = _kaggle()
    username = api.config_values.get(api.CONFIG_NAME_USER)
    return {
        "ok": True,
        "service": "kaggle-gateway",
        "kaggle_authenticated": True,
        "kaggle_user": username,
        "gateway_protected": bool(_secret("GATEWAY_API_KEY")),
    }


@app.post("/api/h3/run")
def h3_run(body: RunRequest, request: Request):
    _check_gateway(request)
    kernel = _validate_ref(body.kernel, "kernel")
    if body.accelerator and not ACC_RE.fullmatch(body.accelerator):
        raise HTTPException(status_code=400, detail="Invalid accelerator")
    api = _kaggle()
    try:
        with tempfile.TemporaryDirectory(prefix="kaggle-gateway-", dir="/tmp") as workdir:
            api.kernels_pull(kernel, path=workdir, metadata=True, quiet=True)
            result = api.kernels_push(
                workdir,
                timeout=str(body.timeout_seconds) if body.timeout_seconds else None,
                acc=body.accelerator,
            )
            error = _first(result, "error", default=None)
            if error:
                raise HTTPException(status_code=502, detail=f"Kaggle kernel push failed: {error}")
            return {
                "ok": True,
                "kernel": kernel,
                "triggered": True,
                "version": _first(result, "version_number", "versionNumber"),
                "url": _first(result, "url"),
                "message": "Kernel version created and execution triggered; poll /api/h3/status for completion.",
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc


@app.get("/api/h3/status")
def h3_status(request: Request, kernel: str = Query(...)):
    _check_gateway(request)
    kernel = _validate_ref(kernel, "kernel")
    api = _kaggle()
    try:
        result = api.kernels_status(kernel)
        return {
            "ok": True,
            "kernel": kernel,
            "status": _enum_or_value(_first(result, "status")),
            "failure_message": _first(result, "failure_message", "failureMessage"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc


@app.get("/api/h3/output-files")
def h3_output_files(
    request: Request,
    kernel: str = Query(...),
    page_token: Optional[str] = Query(default=None),
    page_size: int = Query(default=100, ge=1, le=200),
):
    _check_gateway(request)
    kernel = _validate_ref(kernel, "kernel")
    api = _kaggle()
    try:
        result = api.kernels_list_files(kernel, page_token=page_token, page_size=page_size)
        files = [_file_row(x) for x in (getattr(result, "files", None) or [])]
        return {"ok": True, "kernel": kernel, "files": files, "next_page_token": _next_token(result)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc


@app.get("/api/h3/output")
def h3_output_link(
    request: Request,
    kernel: str = Query(...),
    file: str = Query(...),
):
    _check_gateway(request)
    kernel = _validate_ref(kernel, "kernel")
    file = _validate_file_name(file)
    api = _kaggle()
    owner, slug, _version = api.parse_kernel_string(kernel)
    try:
        page_token = ""
        with api.build_kaggle_client() as kaggle:
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = owner
                req.kernel_slug = slug
                req.page_size = 200
                req.page_token = page_token
                response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                for item in response.files or []:
                    if getattr(item, "file_name", "") == file:
                        url = getattr(item, "url", None)
                        if not url:
                            raise HTTPException(status_code=502, detail="Kaggle did not return a download URL")
                        return RedirectResponse(url=url, status_code=307)
                page_token = getattr(response, "next_page_token", "") or ""
                if not page_token:
                    break
        raise HTTPException(status_code=404, detail="Output file not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc


@app.get("/api/dataset/files")
def dataset_files(
    request: Request,
    dataset: str = Query(...),
    page_token: Optional[str] = Query(default=None),
    page_size: int = Query(default=100, ge=1, le=200),
):
    _check_gateway(request)
    dataset = _validate_ref(dataset, "dataset")
    api = _kaggle()
    try:
        result = api.dataset_list_files(dataset, page_token=page_token, page_size=page_size)
        files = [_file_row(x) for x in (getattr(result, "files", None) or [])]
        return {"ok": True, "dataset": dataset, "files": files, "next_page_token": _next_token(result)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc


@app.get("/api/dataset/download")
def dataset_download(
    request: Request,
    dataset: str = Query(...),
    file: str = Query(...),
):
    _check_gateway(request)
    dataset = _validate_ref(dataset, "dataset")
    file = _validate_file_name(file)
    api = _kaggle()
    try:
        owner, slug, version = api.split_dataset_string(dataset)
        req = ApiDownloadDatasetRequest()
        req.owner_slug = owner
        req.dataset_slug = slug
        req.dataset_version_number = int(version) if version else None
        req.file_name = file
        with api.build_kaggle_client() as kaggle:
            response = kaggle.datasets.dataset_api_client.download_dataset(req)
            url = getattr(response, "url", None)
            if not url:
                raise HTTPException(status_code=502, detail="Kaggle did not return a download URL")
            try:
                response.close()
            except Exception:
                pass
        return RedirectResponse(url=url, status_code=307)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_safe_message(exc)) from exc
