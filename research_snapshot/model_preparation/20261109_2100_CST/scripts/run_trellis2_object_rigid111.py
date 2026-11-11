# Isolated rigid111 adapter; source SHA256: 102520e81bca0f2c5d6d1bbe6683712472abee91948835cdc6eb89cb6f186b78
"""TRELLIS.2 native 1024 cascade, PBR voxel mesh and official GLB export."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

P=Path(__file__).resolve().parents[1]
S=P/'sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd'
parser=argparse.ArgumentParser();parser.add_argument('--sample-id',required=True)
parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--reload-only',action='store_true')
args=parser.parse_args();O=args.run_dir.resolve()
assert O.is_relative_to((P/'runs/trellis2_rigid111_v1').resolve())
os.chdir(S);sys.path.insert(0,str(S))
import torch
torch.empty(1,device='cuda')
import numpy as np
from PIL import Image
import cv2
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
from trellis2.representations import MeshWithVoxel

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
def render(mesh,prefix):
    ex,ins=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,.29670597],[.2,.2],2.,40.)
    hdr=cv2.imread(str(S/'assets/hdri/forest.exr'),cv2.IMREAD_UNCHANGED);assert hdr is not None
    light=EnvMap(torch.tensor(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda'))
    # Official SSAO draws random samples. Isolate a fixed render RNG so
    # independent reload checks measure serialization, not different shadows.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(12345)
        frames=render_utils.render_frames(mesh,ex,ins,options={'resolution':512},envmap=light)
    outputs=[]
    for i in range(2):
        for kind in ['shaded','alpha','base_color']:
            path=O/f'{prefix}_{kind}_{i}.png';Image.fromarray(frames[kind][i]).save(path);outputs.append(path)
    return outputs

if args.reload_only:
    state=json.loads((O/'state.json').read_text());assert state['status']=='generated_pending_reload'
    assert sha(O/'native_voxel_mesh.pt')==state['output_hashes']['native_voxel_mesh.pt']
    packed=torch.load(O/'native_voxel_mesh.pt',map_location='cpu',weights_only=True)
    for k in ['vertices','faces','coords','attrs']:
        assert torch.isfinite(packed[k]).all(),k
        packed[k]=packed[k].cuda()
    packed['layout']={k:slice(*v)for k,v in packed['layout'].items()}
    mesh=MeshWithVoxel(**packed);paths=render(mesh,'reload');views=[]
    for path in paths:
        a=np.array(Image.open(path));b=np.array(Image.open(O/path.name.replace('reload_','native_')))
        diff=np.abs(a.astype(np.int16)-b.astype(np.int16));assert diff.max()<=1 and diff.mean()<=.01
        assert np.isfinite(a).all()
        views.append({'path':str(path),'sha256':sha(path),'max_pixel_difference':int(diff.max()),'nonzero_pixels':int(np.count_nonzero(a)),'sparse_alpha_diagnostic':bool('_alpha_' in path.name and np.count_nonzero(a)<=100)})
    save(O/'reload_validation.json',{'status':'passed_pending_process_exit','views':views,
         'native_checkpoint_sha256':state['output_hashes']['native_voxel_mesh.pt'],
         'render_seed':12345,'scope':'Native output integrity; not common-camera evaluation'})
else:
    O.mkdir(parents=True,exist_ok=False)
    if args.sample_id=='official_example':image_path=S/'assets/example_image/T.png'
    else:
        manifest=P.parents[1]/'inputs/rigid111_image_inputs_v1/manifest.json'
        row=next(x for x in json.loads(manifest.read_text())['objects']if x['sample_id']==args.sample_id)
        image_path=Path(row['image_rgba']);assert sha(image_path)==row['image_rgba_sha256']
    state={'status':'running','stage':'load','pid':os.getpid(),'sample_id':args.sample_id,
           'input':str(image_path),'input_sha256':sha(image_path),'started_at':time.time(),
           'source':str(S),'source_commit':S.name.removeprefix('trellis2-'),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
           'seed':42,'render_seed':12345,'pipeline_type':'1024_cascade','low_vram':True,'max_num_tokens':49152,
           'real_input_count':0 if args.sample_id=='official_example'else 1,'generated_input_count':0,
           'glb_export':{'decimation_target':1000000,'texture_size':4096,'remesh':True,'remesh_band':1,'remesh_project':0},
           'common_camera_qualified':False}
    save(O/'state.json',state)
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
        pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(P/'weights/microsoft--TRELLIS.2-4B'));pipeline.cuda()
        assert pipeline.default_pipeline_type=='1024_cascade' and pipeline.low_vram
        state.update(stage='generate');save(O/'state.json',state)
        mesh=pipeline.run(Image.open(image_path),seed=42)[0]
        mesh.simplify(16777216)
        packed={k:getattr(mesh,k).detach().cpu()for k in ['vertices','faces','coords','attrs']}
        assert all(torch.isfinite(v).all()for v in packed.values())
        packed.update(origin=mesh.origin.tolist(),voxel_size=float(mesh.voxel_size),voxel_shape=list(mesh.voxel_shape),
                      layout={k:[v.start,v.stop,v.step]for k,v in mesh.layout.items()})
        torch.save(packed,O/'native_voxel_mesh.pt')
        state.update(stage='render',vertices=len(mesh.vertices),faces=len(mesh.faces));save(O/'state.json',state)
        render(mesh,'native');state.update(stage='export_glb');save(O/'state.json',state)
        glb=o_voxel.postprocess.to_glb(vertices=mesh.vertices,faces=mesh.faces,attr_volume=mesh.attrs,
             coords=mesh.coords,attr_layout=mesh.layout,voxel_size=mesh.voxel_size,
             aabb=[[-.5,-.5,-.5],[.5,.5,.5]],decimation_target=1000000,texture_size=4096,
             remesh=True,remesh_band=1,remesh_project=0,verbose=True)
        output=O/'native_textured.glb';glb.export(str(output),extension_webp=True)
        state.update(status='generated_pending_reload',stage='finished',output=str(output),output_sha256=sha(output),
                     output_hashes={x.name:sha(x)for x in O.glob('native_*')},peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:state.update(status='failed',reason=repr(error));raise
    finally:state['finished_at']=time.time();save(O/'state.json',state)
