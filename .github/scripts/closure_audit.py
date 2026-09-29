"""Offline dependency closure audit on a disposable CPU runner, not a Kaggle kernel."""
import email, hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

def audit(software, root, node_dir, engine, save, sha):
    software, root = Path(software), Path(root)
    comfy = software / 'ComfyUI'
    wheels = software / 'wheelhouse'
    build = json.loads((software/'build_info.json').read_text())
    inventory = []
    wheel_meta = []
    for p in sorted(software.rglob('*')):
        if not p.is_file(): continue
        inventory.append({'path':str(p.relative_to(software)), 'bytes':p.stat().st_size, 'sha256':sha(p)})
        if p.suffix == '.whl':
            with zipfile.ZipFile(p) as z:
                names=z.namelist(); m=email.message_from_bytes(z.read(next(n for n in names if n.endswith('.dist-info/METADATA'))))
                wheel_meta.append({'file':p.name,'name':m['Name'],'version':m['Version'],
                    'requires_dist':m.get_all('Requires-Dist',[]),'sha256':inventory[-1]['sha256'],
                    'frontend_static_files':sum('/static/' in n for n in names),
                    'bundled_shared_libraries':sum('.so' in n for n in names)})
    save('closure/software_files_sha256.json',inventory)
    save('closure/wheels.lock.json',wheel_meta)
    # Hash-based direct wheel lock avoids resolver drift and index access entirely.
    save('closure/requirements-hashed.lock.txt',''.join(f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n" for w in wheel_meta))
    custom=[]
    for folder in sorted((comfy/'custom_nodes').iterdir()):
        if folder.is_dir():
            record={'path':str(folder.relative_to(comfy)),'git_commit':None,'source_tree_locked_by':'software_files_sha256.json'}
            if (folder/'.git').exists():
                q=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True)
                if q.returncode==0:record['git_commit']=q.stdout.strip()
            custom.append(record)
    save('closure/custom_nodes.lock.json',custom)
    patterns=re.compile(r'git\s+clone|pip\s+install|snapshot_download|hf_hub_download|from_pretrained|load_state_dict_from_url|urlretrieve|https?://')
    hits=[]
    for label,tree in [('comfy',comfy),('multistream',Path(node_dir))]:
        for p in sorted(tree.rglob('*')):
            if not p.is_file() or p.suffix not in ('.py','.js','.json','.toml','.txt','.sh'):continue
            if '.git' in p.parts:continue
            for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if patterns.search(line):hits.append({'path':label+'/'+str(p.relative_to(tree)),'line':i,'text':line[:700]})
    save('closure/network_references_static.json',hits)
    # Fresh environment with no access to system site-packages; all dependencies come from wheels.
    envdir=Path('.preserve/closure_venv').resolve()
    subprocess.run([sys.executable,'-m','venv',str(envdir)],check=True)
    py=envdir/'bin/python'
    logdir=root/'closure';logdir.mkdir(exist_ok=True)
    result={'offline_install':'NOT_RUN','pip_check':'NOT_RUN','cpu_startup':'NOT_RUN',
            'gpu_inference':'NOT_RUN','software_files_hashed':len(inventory),'wheel_count':len(wheel_meta)}
    env=dict(os.environ,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    with (logdir/'offline_install.log').open('w') as log:
        p=subprocess.run([str(py),'-m','pip','install','--no-index','--find-links',str(wheels),
                          '--require-hashes','-r',str(root/'closure/requirements-hashed.lock.txt')],
                         env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    result['offline_install']='PASS' if p.returncode==0 else 'FAIL'
    if p.returncode:
        save('closure/RESULT.json',result)
        print('CLOSURE_INSTALL_FAILURE', (logdir/'offline_install.log').read_text()[-6000:],flush=True)
        return result
    p=subprocess.run([str(py),'-m','pip','check'],env=env,capture_output=True,text=True)
    save('closure/pip_check.log',p.stdout+p.stderr)
    result['pip_check']='PASS' if p.returncode==0 else 'FAIL'
    save('closure/installed.freeze.txt',subprocess.check_output([str(py),'-m','pip','freeze'],env=env,text=True))
    code="import torch,json,sys,platform;print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'platform':platform.platform(),'gpu_available':torch.cuda.is_available()}))"
    p=subprocess.run([str(py),'-c',code],env=env,capture_output=True,text=True)
    save('closure/torch_probe.log',p.stdout+p.stderr)
    result['torch_cpu_import']='PASS' if p.returncode==0 else 'FAIL'
    if p.returncode==0:result['tested_environment']=json.loads(p.stdout.strip().splitlines()[-1])
    target=comfy/'custom_nodes/ComfyUI-H3-MultiStream'
    shutil.copytree(node_dir,target,dirs_exist_ok=True)
    # Run a local-only ComfyUI server in an isolated network namespace if supported.
    probe=Path(__file__).with_name('offline_startup_probe.py').resolve()
    cmd=['sudo','-n','unshare','--net','--','sh','-c','ip link set lo up && exec "$@"','sh',
         str(py),str(probe),str(comfy),str(logdir.resolve()),str((root/'runtime/engine_v24.py').resolve())]
    p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=360)
    save('closure/network_namespace_probe.log',p.stdout+p.stderr)
    if (logdir/'startup_result.json').is_file():
        result['startup_details']=json.loads((logdir/'startup_result.json').read_text())
        result['cpu_startup']='PASS' if result['startup_details'].get('pass') else 'FAIL'
        result['network_namespace']='ISOLATED'
    else:
        result['cpu_startup']='NOT_VERIFIED'
        result['network_namespace']='UNAVAILABLE_OR_STARTUP_FAILED'
    result['comfy_build_info']=build
    result['extra_dependencies']='MultiStream declares no dependencies beyond ComfyUI; HEIC wheels preserved separately.'
    result['closure_scope']='V24 workflow software; optional unrelated node download paths are inventoried, not executed.'
    save('closure/RESULT.json',result)
    print('CLOSURE_AUDIT',json.dumps(result),flush=True)
    return result
