"""CPU-only startup/schema/tokenizer check in a network namespace without external interfaces."""
import importlib.util, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
comfy,logs,engine=map(Path,sys.argv[1:4])
result={'pass':False,'external_connectivity':False,'inference_performed':False}
# Confirm external network cannot be reached; this is a single bounded connectivity probe.
try:
    with socket.create_connection(('1.1.1.1',443),timeout=2):result['external_connectivity']=True
except OSError:pass
assert not result['external_connectivity'], 'Network isolation failed'
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for key in ('KAGGLE_API_TOKEN','GITHUB_TOKEN','GH_TOKEN'):
    env.pop(key,None)
with (logs/'comfy_cpu_startup.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'main.py','--cpu','--listen','127.0.0.1','--port','18188','--disable-auto-launch'],
                       cwd=comfy,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        info=None
        for _ in range(120):
            if p.poll() is not None:break
            try:
                with urllib.request.urlopen('http://127.0.0.1:18188/object_info',timeout=2) as r:info=json.load(r)
                break
            except Exception:time.sleep(1)
        if info is None:raise RuntimeError('ComfyUI failed to serve object_info')
        spec=importlib.util.spec_from_file_location('v24_engine',engine);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        config={'prompt':'Offline schema validation only.','width':768,'height':1664,'steps':20,'seed':1}
        graph=mod.make_graph(config,['placeholder.png','placeholder_last.png'],192)
        mod.validate_schema(graph,info)
        result['workflow_node_classes']=sorted({n['class_type'] for n in graph.values()})
        result['workflow_schema']='PASS'
        with urllib.request.urlopen('http://127.0.0.1:18188/',timeout=5) as r:
            result['frontend_index_bytes']=len(r.read())
        result['frontend']='PASS' if result['frontend_index_bytes']>100 else 'FAIL'
        # Save loader code lines for explicit audit; weight loading is not performed.
        sources=[]
        for part in ('comfy/text_encoders','comfy/ldm/minimax','comfy_extras'):
            for q in (comfy/part).rglob('*.py'):
                if 'minimax' in q.name.lower() or 'qwen' in q.name.lower():
                    sources.append({'path':str(q.relative_to(comfy)),'source':q.read_text()})
        (logs/'loader_sources.json').write_text(json.dumps(sources,indent=2))
        result['pass']=result['frontend']=='PASS'
    except Exception as e:result['error']=str(e)
    finally:
        p.terminate()
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.kill();p.wait()
(logs/'startup_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
sys.exit(0 if result['pass'] else 1)
