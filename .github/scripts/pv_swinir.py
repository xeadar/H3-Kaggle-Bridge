import sys, subprocess, json, zipfile, time
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', 'pillow-heif', 'timm'], check=True)
from pillow_heif import register_heif_opener
register_heif_opener()
from PIL import Image
import numpy as np
import torch

TARGET_SHORT = 2160
TILE = 320
OVERLAP = 32
SCALE = 2
WINDOW = 8
EXCLUDE = {'IMG_2446.HEIC', 'IMG_2446.heic', 'IMG_2447.HEIC', 'IMG_2447.heic'}

work = Path('/kaggle/working')
input_roots = [p for p in Path('/kaggle/input').iterdir() if p.is_dir() and p.name.startswith('pv-swinir-temp-')]
if len(input_roots) != 1:
    raise SystemExit(f'Expected exactly one temporary input Dataset, found {[p.name for p in input_roots]}')
src_root = input_roots[0] / 'Pv'
out_root = work / 'Pv_SwinIR_2160'
out_root.mkdir(parents=True, exist_ok=True)
if not src_root.is_dir():
    raise SystemExit(f'Input folder missing: {src_root}')

repo = work / 'SwinIR'
if not repo.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/JingyunLiang/SwinIR.git', str(repo)], check=True)
sys.path.insert(0, str(repo))
from models.network_swinir import SwinIR

model_path = work / '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
if not model_path.exists():
    subprocess.run([
        'curl', '-L', '--fail', '--retry', '5', '--retry-delay', '2',
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth',
        '-o', str(model_path)
    ], check=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemExit('GPU not available')
print('DEVICE', torch.cuda.get_device_name(0), flush=True)

model = SwinIR(
    upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
    depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
    upsampler='nearest+conv', resi_connection='1conv'
)
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['params_ema'], strict=True)
model.eval().to(device)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


def tile_forward(x):
    b, c, h, w = x.shape
    tile = min(TILE, h, w)
    tile -= tile % WINDOW
    if tile <= 0:
        tile = WINDOW
    overlap = min(OVERLAP, tile - WINDOW)
    stride = max(WINDOW, tile - overlap)
    h_idx = list(range(0, max(h - tile, 0) + 1, stride))
    w_idx = list(range(0, max(w - tile, 0) + 1, stride))
    if not h_idx or h_idx[-1] != h - tile:
        h_idx.append(max(h - tile, 0))
    if not w_idx or w_idx[-1] != w - tile:
        w_idx.append(max(w - tile, 0))
    accum = torch.zeros((b, c, h * SCALE, w * SCALE), dtype=torch.float32, device='cpu')
    weight = torch.zeros_like(accum)
    total = len(h_idx) * len(w_idx)
    done = 0
    for hi in h_idx:
        for wi in w_idx:
            inp = x[..., hi:hi + tile, wi:wi + tile]
            out = model(inp).float().cpu()
            oh, ow = hi * SCALE, wi * SCALE
            accum[..., oh:oh + tile * SCALE, ow:ow + tile * SCALE] += out
            weight[..., oh:oh + tile * SCALE, ow:ow + tile * SCALE] += 1.0
            done += 1
            if done % 8 == 0 or done == total:
                print(f'    tiles {done}/{total}', flush=True)
    return accum / weight.clamp_min(1.0)


def run_image(path):
    t0 = time.time()
    im = Image.open(path).convert('RGB')
    ow, oh = im.size
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    pad_h = (WINDOW - oh % WINDOW) % WINDOW
    pad_w = (WINDOW - ow % WINDOW) % WINDOW
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    x = x.to(device)
    y = tile_forward(x)
    y = y[..., :oh * SCALE, :ow * SCALE].clamp_(0, 1)
    out_arr = (y.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out = Image.fromarray(out_arr, 'RGB')
    sw, sh = out.size
    factor = TARGET_SHORT / min(sw, sh)
    final_size = (round(sw * factor), round(sh * factor))
    if out.size != final_size:
        out = out.resize(final_size, Image.Resampling.LANCZOS)
    if path.suffix.lower() == '.png':
        out_name = path.stem + '_SwinIR_2160.png'
        out.save(out_root / out_name, format='PNG', compress_level=6)
    else:
        out_name = path.stem + '_SwinIR_2160.jpg'
        out.save(out_root / out_name, format='JPEG', quality=97, subsampling=0, optimize=True)
    elapsed = time.time() - t0
    torch.cuda.empty_cache()
    return {
        'source': path.name,
        'source_size': [ow, oh],
        'swinir_model': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN',
        'swinir_scale': 2,
        'final_size': list(final_size),
        'output': out_name,
        'seconds': round(elapsed, 2)
    }


files = sorted(p for p in src_root.iterdir() if p.is_file() and p.name not in EXCLUDE and not p.name.startswith('._'))
if len(files) != 12:
    raise SystemExit(f'Expected 12 images after exclusions, found {len(files)}: {[p.name for p in files]}')
report = []
print('FILES', len(files), [p.name for p in files], flush=True)
for i, path in enumerate(files, 1):
    print(f'[{i}/{len(files)}] START {path.name}', flush=True)
    rec = run_image(path)
    report.append(rec)
    (out_root / '_process_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[{i}/{len(files)}] DONE {rec}', flush=True)

zip_path = work / 'Pv_SwinIR_2160.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(out_root.iterdir()):
        zf.write(p, arcname=p.name)
print('FINAL_ZIP', zip_path, zip_path.stat().st_size, flush=True)
