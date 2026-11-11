"""Frozen vehicle adapter: same native settings as validated frozen pilot."""
import os
import sys
import json
import time
import hashlib
from pathlib import Path

P = Path(__file__).resolve().parents[1]
S = P/'sources/Hunyuan3D-2.1-82920d643c0dc2f7bfd7255f45f62d386edfe60c'
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--sample-id',required=True)
parser.add_argument('--run-dir',type=Path,required=True)
args=parser.parse_args()
manifest=P.parents[1]/'inputs/vehicle432_image_inputs_v1/manifest.json'
row=next(x for x in json.loads(manifest.read_text())['objects'] if x['sample_id']==args.sample_id)
input_record={'sample_id':row['sample_id'],'source_image':row['image_rgba'],'source_image_sha256':row['image_rgba_sha256']}
INPUT=Path(input_record['source_image'])
assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==input_record['source_image_sha256']
assert json.loads((P/'runs/hunyuan21_official_demo_v1/state.json').read_text())['official_example_validated']
O = args.run_dir.resolve()
assert O.is_relative_to((P/'runs/hunyuan21_vehicle432_v1').resolve())
resume_paint = False
O.mkdir(parents=True, exist_ok=resume_paint)
sys.path[:0] = [str(S), str(S/'hy3dshape'), str(S/'hy3dpaint')]
os.chdir(S)
import torch
# Establish the CUDA context immediately so the AH idle-GPU scheduler sees it.
torch.empty(1, device='cuda')
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from torchvision_fix import apply_fix
assert apply_fix(), 'Official torchvision compatibility helper failed'

state = {'status':'running','stage':'shape','pid':os.getpid(),'started_at':time.time(),
         'source_commit':S.name.removeprefix('Hunyuan3D-2.1-'),
         'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'), 'official_example_validated':False, 'sample_id':input_record['sample_id'], 'shape_seed':42, 'texture_seed':0,
         'input':str(INPUT), 'input_manifest':str(manifest), 'input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
         'input_sha256':hashlib.sha256((INPUT).read_bytes()).hexdigest(),
         'settings':'Frozen real RGBA input; native defaults, fp16 shape seed 42, 6 texture views 512 texture seed 0; no OOM fallback.'}


def save():
    temp = O/'state.tmp'
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(O/'state.json')


save()
try:
    if resume_paint:
        import trimesh
        previous = json.loads((O/'state.before_torchvision_fix.json').read_text())
        assert previous['status']=='failed' and previous['stage']=='paint'
        assert previous['input_sha256']==state['input_sha256']
        expected = (O/'shape_resume_sha256.txt').read_text().strip()
        assert hashlib.sha256((O/'demo.glb').read_bytes()).hexdigest()==expected
        mesh = trimesh.load(O/'demo.glb', force='mesh')
        state.update(reused_shape_sha256=expected, official_torchvision_fix_applied=True)
    else:
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(P/'weights/tencent--Hunyuan3D-2.1'))
        image = Image.open(INPUT).convert('RGBA')
        mesh = pipeline_shapegen(image=image,generator=torch.Generator(device='cuda').manual_seed(42))[0]
        mesh.export(O/'demo.glb')
    state.update(stage='paint',shape_vertices=len(mesh.vertices),shape_faces=len(mesh.faces)); save()
    conf = Hunyuan3DPaintConfig(6, 512)
    conf.realesrgan_ckpt_path = str(P/'weights/auxiliary/RealESRGAN_x4plus.pth')
    conf.multiview_cfg_path = str(S/'hy3dpaint/cfgs/hunyuan-paint-pbr.yaml')
    conf.custom_pipeline = str(S/'hy3dpaint/hunyuanpaintpbr')
    conf.dino_ckpt_path = str(P/'weights/facebook--dinov2-giant')
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    result = paint_pipeline(mesh_path=str(O/'demo.glb'), image_path=str(INPUT),
                           output_mesh_path=str(O/'demo_textured.glb'))
    import trimesh
    output = Path(result)
    assert output.is_file()
    reloaded = trimesh.load(output, force='scene')
    assert reloaded.geometry and sum(len(x.vertices) for x in reloaded.geometry.values()) > 0
    state.update(status='generated_and_mesh_reload_passed_pending_render',stage='finished',
                 output=str(output),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 official_example_validated=False,peak_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as error:
    state.update(status='failed',reason=repr(error))
    raise
finally:
    state['finished_at']=time.time();save()
