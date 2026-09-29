"""Preserve V24 runtime dependencies only. Never launches or edits a Kaggle kernel."""
import ast, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from closure_audit import audit as audit_closure
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

KERNEL = "sita2ksitas/garden-tales-v20-h3-pv-s03"
DEST = "sita2ksitas/h3-v24-runtime-dependencies"
EXPECTED_SOURCE_SHA = "51845eb6577293e4019732ecaa8e2773c5b2eae8c62ebc5b0a0c27147782fdfa"
B_COMMIT = "4e4381ff2cdb061a4cd0065af5b9c3679ded379a"
root = Path(".preserve/payload")
root.mkdir(parents=True, exist_ok=True)
api = KaggleApi(); api.authenticate()
def run(args, **kw):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True, text=True, **kw).stdout
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def save(path, value):
    p=root/path; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    return p
def literal_assignment(text,name):
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and any(getattr(t,"id","")==name for t in n.targets):
            return ast.literal_eval(n.value)
    raise RuntimeError("Missing assignment: "+name)
req=ApiGetKernelRequest(); req.user_name=KERNEL.split("/")[0]; req.kernel_slug=KERNEL.split("/")[1]
with api.build_kaggle_client() as client: response=client.kernels.kernels_api_client.get_kernel(req)
meta=response.metadata.to_dict(); source=response.blob.source
assert meta["currentVersionNumber"]==24, "Source changed: re-audit first"
assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED_SOURCE_SHA, "Source changed: re-audit first"
assert not meta.get("kernelDataSources"), "Unexpected kernel dependencies: re-audit first"
nb=json.loads(source); cells=nb["cells"]
texts=["".join(c["source"]) if isinstance(c.get("source"),list) else c.get("source","") for c in cells]
engine_cell=next(t for t in texts if "ENGINE_SOURCE =" in t)
engine=literal_assignment(engine_cell,"ENGINE_SOURCE")
config=literal_assignment(next(t for t in texts if "CONFIG = {" in t),"CONFIG")
saved_config={k:v for k,v in config.items() if k not in ("prompt","first_frame","last_frame")}
safe_config=dict(saved_config,mode="prepare",prompt="",first_frame="",last_frame="")
save("runtime/engine_v24.py",engine)
save("runtime/engine_cell_v24.py",engine_cell)
save("runtime/config_template.json",safe_config)
save("runtime/original_numeric_settings.json",saved_config)
save("runtime/source_provenance.json",{
 "kernel_ref":KERNEL,"kernel_id":meta["id"],"version":24,"source_sha256":EXPECTED_SOURCE_SHA,
 "title":meta["title"],"docker_image":meta.get("dockerImage"),"machine_shape":meta.get("machineShape"),
 "kernel_sources":[],"dataset_sources":[d for d in meta.get("datasetDataSources",[]) if d],
 "privacy":"Original prompt, input media names, cell outputs and obsolete archive cell are excluded."})
# A source snapshot, deliberately without a generation cell or old archive destinations.
snapshot={"nbformat":4,"nbformat_minor":5,"metadata":{},
 "cells":[
 {"cell_type":"markdown","metadata":{},"source":"# V24 runtime source snapshot\nNo generation or archiving is triggered. Supply new approved input assets in a future notebook."},
 {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":"CONFIG = "+repr(safe_config)},
 {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":engine_cell}]}
save("runtime/V24_core_source_snapshot.ipynb",snapshot)
# Preserve the T4 helper as a real file, rather than only a string inside the engine.
helper=None
for node in ast.walk(ast.parse(engine)):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="write_text" and getattr(node.func.value,"id","")=="helper":
        helper=ast.literal_eval(node.args[0])
assert helper
save("patches/garden_t4_attention.py",helper)
save("patches/README.md","The exact T4 helper and split.py replacement logic are preserved in runtime/engine_v24.py. Do not replace this with generic attention flags. GPU numerical validation remains a later runtime step.\n")
# Pin upstream node source. Keep expanded source and a shallow Git metadata archive separately.
repo=Path(".preserve/upstream")
run(["git","init",repo])
run(["git","-C",repo,"remote","add","origin","https://github.com/martonsagi/Comfy-H3-MultiStream.git"])
run(["git","-C",repo,"fetch","--depth=1","origin",B_COMMIT])
run(["git","-C",repo,"checkout","--detach","FETCH_HEAD"])
assert run(["git","-C",repo,"rev-parse","HEAD"]).strip()==B_COMMIT
node_dir=root/"custom_nodes/ComfyUI-H3-MultiStream"
shutil.copytree(repo,node_dir,ignore=shutil.ignore_patterns(".git"))
save("custom_nodes/MULTISTREAM_COMMIT.txt",B_COMMIT+"\n")
git_zip=root/"custom_nodes/multistream_git_metadata.zip.bin"
with zipfile.ZipFile(git_zip,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted((repo/".git").rglob("*")):
        z.write(p,p.relative_to(repo))
# Check the archived git metadata really satisfies V24's git checkout/show calls offline.
probe=Path(".preserve/git_restore_check");shutil.copytree(node_dir,probe)
with zipfile.ZipFile(git_zip) as z:z.extractall(probe)
assert run(["git","-C",probe,"rev-parse","HEAD"]).strip()==B_COMMIT
split=run(["git","-C",probe,"show",B_COMMIT+":multistream/split.py"])
assert split==(node_dir/"multistream/split.py").read_text()
# Existing large Datasets stay in place. Inspect the software once, but do not duplicate it.
print('DOWNLOADING_EXISTING_SOFTWARE_FOR_CLOSURE_AUDIT',flush=True)
software_dataset=Path(kagglehub.dataset_download('sita2ksitas/h3-comfyui-offline-software/versions/1'))
software=software_dataset/'software'
assert (software/'ComfyUI/main.py').is_file()
print('EXISTING_SOFTWARE_READY',flush=True)
for ref,path,out in [
 ("sita2ksitas/h3-comfyui-offline-software/versions/1","software/build_info.json","existing/build_info.json"),
 ("sita2ksitas/h3-comfyui-offline-software/versions/1","software/requirements.lock.txt","existing/requirements.lock.txt"),
 ("sita2ksitas/minimax-h3-comfyui-models/versions/2","model_manifest.json","existing/model_manifest.json")]:
    p=Path(kagglehub.dataset_download(ref,path=path))
    target=root/out;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
closure=audit_closure(software,root,node_dir,engine,save,sha)
save('VERSION_LOCK.json',{
 'source_kernel_version':24,'source_sha256':EXPECTED_SOURCE_SHA,
 'comfy_build':json.loads((root/'existing/build_info.json').read_text()),
 'multistream_commit':B_COMMIT,
 'python_notebook_metadata':nb.get('metadata',{}).get('language_info',{}).get('version'),
 'kaggle_docker_image':meta.get('dockerImage'),
 'tested_cpu_environment':closure.get('tested_environment'),
 'wheel_lock':'closure/wheels.lock.json','hash_enforced_install_lock':'closure/requirements-hashed.lock.txt',
 'software_tree_sha256_inventory':'closure/software_files_sha256.json',
 'platform_boundary':'Kaggle host NVIDIA driver is not part of a Dataset; GPU compatibility must be checked on rebuild.',
 'baseline_status':'V24 source preservation; fresh GPU inference acceptance is not performed.'})
# Original V24 did not pin pillow-heif. Resolve now, save wheels and lock exact versions.
wheel_dir=root/"heic_wheels";wheel_dir.mkdir()
run([sys.executable,"-m","pip","download","--only-binary=:all:","--dest",wheel_dir,"pillow-heif","pillow==12.3.0"],timeout=300)
heic_py=Path(".preserve/heic_test_venv/bin/python")
run([sys.executable,"-m","venv",heic_py.parent.parent])
run([heic_py,"-m","pip","install","--no-index","--find-links",wheel_dir,"pillow-heif","pillow==12.3.0"],timeout=180)
versions=run([heic_py,"-m","pip","freeze"])
save("heic_wheels/requirements-heic.lock.txt",versions)
test="from PIL import Image; from pillow_heif import register_heif_opener; from pathlib import Path; register_heif_opener(); p=Path('.preserve/heic_fixture.heic'); Image.new('RGB',(32,32),(80,120,160)).save(p); im=Image.open(p); im.load(); assert im.size==(32,32); im.convert('RGB').save('.preserve/heic_fixture.png'); print('OFFLINE_HEIC_ROUNDTRIP_PASS')"
heic_result=run([heic_py,"-c",test])
assert "PASS" in heic_result
for p in (root/"runtime").glob("*.py"):compile(p.read_text(),str(p),"exec")
compile(helper,"garden_t4_attention.py","exec")
# Build complete file inventories without downloading 42 GB of existing model weights.
inventories={}
for ref in ["sita2ksitas/h3-comfyui-offline-software","sita2ksitas/minimax-h3-comfyui-models"]:
    files=[];token=None
    while True:
        page=api.dataset_list_files(ref,page_token=token,page_size=100)
        files.extend(page.to_dict().get("datasetFiles",[]))
        token=getattr(page,"next_page_token",None)
        if not token:break
    inventories[ref]=files
save("existing/dataset_file_inventories.json",inventories)
models=literal_assignment(engine,"MODELS")
listed={Path(f["name"]).name:f.get("totalBytes",0) for f in inventories["sita2ksitas/minimax-h3-comfyui-models"]}
for rel,(size,digest) in models.items():assert listed.get(Path(rel).name)==size
save("MODEL_REQUIREMENTS.json",{rel:{"bytes":size,"sha256_expected":digest} for rel,(size,digest) in models.items()})
save("VALIDATION.json",{
 "source_version":24,"source_hash_match":True,"old_notebook_dependencies":0,
 "expanded_node_source_preserved":True,"git_metadata_offline_check":"PASS",
 "python_syntax":"PASS","heic_offline_roundtrip":"PASS","heic_lock":versions.strip().splitlines(),
 "model_file_sizes_match":True,"existing_model_bytes_rehashed":False,
 "gpu_started":False,"inference_test_performed":False})
save("README.md",'''# H3 V24 Runtime Dependencies

Private preservation package for a future fresh Notebook. No videos or source-frame assets are included.

## Authoritative source
Kaggle kernel sita2ksitas/garden-tales-v20-h3-pv-s03, currentVersionNumber 24.
The title still says v20. Runtime and settings were retrieved from the API and hashed.

## Required Dataset inputs
1. sita2ksitas/h3-comfyui-offline-software, version 1: expanded software/ComfyUI, wheelhouse, requirements.lock.txt.
2. sita2ksitas/minimax-h3-comfyui-models, version 2: four large model files. Already present; not duplicated here.
3. sita2ksitas/h3-v24-runtime-dependencies: this package.
4. Future user-selected input frames: a new input Dataset or explicit upload, not the deleted old assets.

There are ZERO mounted old Notebook outputs in V24 and no kernel_output download in the engine.
The Version 4 line still printed in prepare mode is stale text, not a live dependency.

## What this package preserves
- Exact engine code, engine cell, clean source snapshot, numerical settings, source provenance.
- Pinned MultiStream source at 4e4381ff2cdb061a4cd0065af5b9c3679ded379a, LICENSE and NOTICE.
- Exact Garden Tales T4 dense-attention helper, with patch logic retained in the engine.
- HEIC decoder wheels and Pillow 12.3.0 for CPython 3.12 Linux x86_64, resolved and locked during preservation.
- Existing package/model manifests and all filenames/sizes.

The upstream MultiStream source is fully expanded in custom_nodes/ComfyUI-H3-MultiStream.
The small separate multistream_git_metadata.zip.bin restores its .git directory for V24's git checkout/show operations.
The .bin suffix prevents Kaggle's automatic archive expansion; read it with Python zipfile.ZipFile.
This is NOT the old 3.8 GB comfyui_offline.tar requirement.

## Future Notebook integration
Copy the node source into the writable ComfyUI custom_nodes directory before engine startup.
Extract the supplied Git metadata ZIP at the node directory. The exact V24 git checkout/show calls then work offline.
Install HEIC wheels into the interpreter performing frame decoding using --no-index --find-links.
Keep the two large existing Datasets mounted. Use manifest discovery rather than assume mount directory names.
Configure fresh authorized frames and prompt. Do not run the source snapshot as a finished new-generation Notebook.
A new archiving destination is needed if publishing outputs; deleted Garden Tales PV is NOT recreated here.
The former archive cell was excluded from the clean snapshot, together with old prompts, input media names and outputs.

## Known V24 limitations that the new Notebook must address
- 768 x 1664 is fixed and exactly 9:19.5; this is not general automatic aspect-ratio calculation.
- 8 seconds, 20 steps, 24 fps, 192 native frames; shares 0.85/1.15.
- The mode named prepare still runs the T4 attention GPU self-test: it is not a fully CPU-only preflight.
- The original HEIC installation had no version pin; these newly pinned decoder wheels pass a CPU roundtrip,
  but their version is not asserted to equal the historical V24 pip resolution.
- Kaggle base image, Python 3.12, CUDA drivers and two compatible T4 GPUs remain platform requirements.
- This preservation does not verify video inference and does not start any GPU.
- Do not delete the original Notebook until the future replacement has passed its own acceptance test.

All third-party software remains under its included upstream license. No model license is changed.
''')
save('closure/README.md','''# Offline closure and version lock

VERSION_LOCK.json records ComfyUI build provenance, MultiStream commit, Python metadata,
Kaggle base-image digest, Torch/CUDA probe results and links to hash-locked wheels.
The full existing software tree has been read and hashed, without duplicating its 3.79 GB here.
The fresh CPU environment uses only these local wheels: --no-index --require-hashes.
CPU ComfyUI startup and V24 node-schema/frontend checks are attempted in a network namespace
without external interfaces. RESULT.json records the actual result, not an inference claim.

network_references_static.json includes optional APIs and unrelated node download paths;
their mere presence does not prove that the V24 workflow calls them.
loader_sources.json, when produced, captures the H3/Qwen loader code for review.
Weights are not loaded and CUDA kernels are not tested during this CPU-only preservation.
The exact NVIDIA host driver cannot be vendored in a Dataset.

Future integration must preload the supplied MultiStream source, install HEIC from its local
wheels, mount all model inputs ahead of time, disable online fallback and fail clearly when
an input is missing. The exact preserved V24 engine itself has NOT been silently modified.
''')
files=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(root.rglob("*")) if p.is_file()]
manifest={"schema":1,"source_kernel":KERNEL,"source_version":24,"source_sha256":EXPECTED_SOURCE_SHA,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"files":files}
save("MANIFEST.json",manifest)
print("PACKAGE_READY",json.dumps({"files":len(files)+1,"bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}))
kagglehub.dataset_upload(DEST,str(root),version_notes="Preserve V24 software dependencies; no generation, no media assets")
# Full byte-for-byte verification from the newly persisted Dataset.
downloaded=None
for attempt in range(8):
    try:
        downloaded=Path(kagglehub.dataset_download(DEST,force_download=True))
        if (downloaded/"MANIFEST.json").is_file() and json.loads((downloaded/"MANIFEST.json").read_text())==manifest:break
    except Exception:
        if attempt==7:raise
    time.sleep(10)
assert downloaded is not None and (downloaded/"MANIFEST.json").is_file()
remote=json.loads((downloaded/"MANIFEST.json").read_text())
assert remote==manifest
for f in manifest["files"]:
    p=downloaded/f["path"]
    assert p.is_file() and p.stat().st_size==f["bytes"] and sha(p)==f["sha256"], f["path"]
# Confirm visibility using dataset metadata.
info_dir=Path(".preserve/visibility");info_dir.mkdir()
run(["kaggle","datasets","metadata",DEST,"-p",info_dir])
vis=json.loads((info_dir/"dataset-metadata.json").read_text())
receipt={"dataset":DEST,"url":"https://www.kaggle.com/datasets/"+DEST,"source_version":24,
 "old_notebook_dependencies":0,"file_count":len(files)+1,"payload_bytes":manifest["total_bytes"],
 "sha256_readback":"PASS","gpu_started":False,"new_notebook_created":False,
 "closure":closure,
 "dataset_metadata":{k:vis.get(k) for k in ("id","title","isPrivate","is_private")}}
Path("preservation-receipt.json").write_text(json.dumps(receipt,indent=2))
print("PRESERVATION_COMPLETE",json.dumps(receipt))
