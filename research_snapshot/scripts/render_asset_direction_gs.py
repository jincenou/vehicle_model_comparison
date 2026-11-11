"""Visualization-only semantic heading alignment. Never modifies asset or metric inputs."""
import os,json,hashlib,importlib.util,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization
R=Path(__file__).resolve().parents[1];P=R/'model_preparation/20261109_2100_CST';O=R/'reports/visual_aligned_20261110'
parser=argparse.ArgumentParser();parser.add_argument('--only-method',default=None);args=parser.parse_args()
torch.set_num_threads(4);torch.empty(1,device='cuda')
reg={x['sample_id']:x for x in json.loads((R/'configs/metric53_registry.json').read_text())['samples']}
spec=importlib.util.spec_from_file_location('frozen_ply',Path(('/opt/rigid-benchmark/assetbench/repos/ah/asset_harvester/tokengs/ply_io.py'.replace('/opt/rigid-benchmark/assetbench', __import__("os").environ.get('RIGID_ASSETBENCH', '/opt/rigid-benchmark/assetbench')).replace('/opt/rigid-benchmark/project', __import__("os").environ.get('RIGID_PROJECT_ROOT', '/opt/rigid-benchmark/project')))));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for sid in ['scene-0064_t000961','scene-0063_t000792']:
 for method in ['ours','ah','trellis_original','recon02']:
  if args.only_method and method!=args.only_method:continue
  out=O/sid/method;out.mkdir(parents=True,exist_ok=False)
  if method in ['ours','ah']:
   asset=reg[sid]['assets'][method];path=Path(asset['ply']);assert sha(path)==asset['sha256'];axes=np.array(asset['native_to_zup'],float)
   if method=='ah':axes=np.diag([-1,-1,1])@axes
  else:
   path=P/f'runs/{method}_vehicle432_v1/{sid}/native_gaussians.ply'
   state=json.loads((path.parent/'state.json').read_text());assert sha(path)==state['output_hashes'][path.name]
   native_to_ply=np.array([[1,0,0],[0,0,-1],[0,1,0]])
   native_to_semantic=np.array([[0,-1,0],[1,0,0],[0,0,1]])
   axes=native_to_semantic@native_to_ply.T if method=='trellis_original' else native_to_semantic
  assert np.allclose(axes@axes.T,np.eye(3))and np.linalg.det(axes)>0
  v=module.read_ply(str(path));assert not any(n.startswith('f_rest_')for n in v.keys())
  fields=lambda names:np.stack([v[n]for n in names],axis=-1)
  xyz=fields(['x','y','z']);q=fields(['rot_0','rot_1','rot_2','rot_3']);q/=np.linalg.norm(q,axis=-1,keepdims=True)
  scales=np.exp(fields(['scale_0','scale_1','scale_2']));opacity=1/(1+np.exp(-v['opacity']));rgb=np.clip(fields(['f_dc_0','f_dc_1','f_dc_2'])*.28209479177387814+.5,0,1)
  common=xyz@axes.T;lo,hi=np.quantile(common,[.005,.995],axis=0);center=(lo+hi)/2;length=hi[0]-lo[0]
  assert length>0
  tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda').contiguous()
  arrays=[tensor(x)for x in [xyz,q,scales,opacity,rgb]]
  f=256/np.tan(np.deg2rad(40)/2);K=np.array([[f,0,256],[0,f,256],[0,0,1.]])
  records=[]
  for name,az in [('front_45',45),('side_90',90),('rear_135',135)]:
   az,el=np.deg2rad([az,12]);direction=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
   forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);rot=np.stack([right,np.cross(forward,right),forward])
   view=np.eye(4);view[:3,:3]=rot@axes;view[:3,3]=-rot@(center+1.8*length*direction)
   with torch.inference_mode():im,alpha,_=rasterization(*arrays,tensor(view)[None],tensor(K)[None],512,512,backgrounds=torch.ones((1,3),device='cuda'),near_plane=.01*length,far_plane=20*length,packed=False)
   assert torch.isfinite(im).all();Image.fromarray((im[0,:,:,:3].clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(out/(name+'.png'))
   records.append(dict(name=name,w2c_native=view.tolist(),K=K.tolist()))
  (out/'view_manifest.json').write_text(json.dumps(dict(sample_id=sid,method=method,asset=str(path),asset_sha256=sha(path),native_to_semantic=axes.tolist(),front='+X',left='+Y',up='+Z',centering='Gaussian center .005/.995 quantile box',length=float(length),camera_distance_over_length=1.8,vertical_fov_degrees=40,elevation_degrees=12,views=records,scope='heading-normalized qualitative display; NOT physical registration or metric camera',renderer='gsplat degree0; comparison preview only'),indent=2)+'\n')
  del arrays;torch.cuda.empty_cache();print(sid,method,'done',flush=True)
