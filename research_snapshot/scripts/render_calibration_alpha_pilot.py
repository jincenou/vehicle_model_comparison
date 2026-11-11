"""Native alpha-only orthographic measurement pilot; never a formal score adapter."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];MODULES=R/'evaluation_nuscenes_v1/scripts'
if not MODULES.exists():MODULES=R.parent/'evaluation'
sys.path.insert(0,str(MODULES))
from alpha_extent_measurement import rig,measure_view,combine,convergence
from spatial_calibration import length_conditioned


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()


def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample-id',default='scene-0064_t000637');ap.add_argument('--method',required=True,choices=['ours','ah','trellis_original','recon02','trellis2','hunyuan21','fixture_mesh','fixture_gaussian']);ap.add_argument('--resolutions',type=int,nargs=2,default=[512,1024]);a=ap.parse_args()
 assert a.resolutions[1]==2*a.resolutions[0] and a.resolutions[0]>=128
 out=R/'exports/calibration_alpha_pilot_v1'/a.sample_id/a.method;out.mkdir(parents=True,exist_ok=False)
 state=dict(status='running',method=a.method,sample_id=a.sample_id,pid=os.getpid(),started_at=time.time(),formal_metrics_qualified=False,raw_asset_modified=False,resolutions=a.resolutions,threshold='uint8_alpha > 127',camera_coordinates='pixel edges; centres i+.5',views=[])
 state['code_sha256']={str(p):sha(p) for p in [Path(__file__),MODULES/'alpha_extent_measurement.py',MODULES/'spatial_calibration.py']};save(out/'state.json',state)
 try:
  fixture=a.method.startswith('fixture_');mesh=a.method in ['trellis2','hunyuan21','fixture_mesh'];axes=np.eye(3);asset=None
  if not fixture:
   dp=R/'exports/asset_orientation_final_v1'/a.method/a.sample_id/'orientation_render.json';display=json.loads(dp.read_text());asset=Path(display['asset']);assert sha(asset)==display['asset_sha256']
   hp=Path(display['heading_decision']);assert sha(hp)==display['heading_decision_sha256'];heading=json.loads(hp.read_text());assert heading['asset_sha256']==display['asset_sha256']
   axes=np.asarray(heading['native_to_semantic']);state.update(asset=str(asset),asset_sha256=sha(asset),heading=str(hp),heading_sha256=sha(hp))
  if mesh:
   import bpy
   from mathutils import Matrix
   bpy.ops.wm.read_factory_settings(use_empty=True)
   if fixture:
    bpy.ops.mesh.primitive_cube_add(size=1);bpy.context.object.scale=(2,1,.5)
   else:bpy.ops.import_scene.gltf(filepath=str(asset))
   meshes=[x for x in bpy.context.scene.objects if x.type=='MESH'];assert meshes
   # Flush scale/import changes before reading matrix_world. Capture every world
   # matrix before assigning any, so parent-child imports retain their transforms.
   bpy.context.view_layer.update()
   worlds={obj.name:obj.matrix_world.copy() for obj in meshes}
   A=np.eye(4);A[:3,:3]=axes
   def depth(obj):
    count=0
    while obj.parent is not None:count+=1;obj=obj.parent
    return count
   for obj in sorted(meshes,key=depth):
    obj.matrix_world=Matrix(A.tolist())@worlds[obj.name]
    bpy.context.view_layer.update()
   bpy.context.view_layer.update()
   corners=np.array([tuple(obj.matrix_world@__import__('mathutils').Vector(v)) for obj in meshes for v in obj.bound_box]);low=corners.min(0);high=corners.max(0)
   state['framing_rule']='All native mesh bounding corners; framing only, not measured extent'
   scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=42
   scene.render.threads_mode='FIXED';scene.render.threads=2;scene.render.resolution_percentage=100;scene.render.film_transparent=True
   scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard'
   cd=bpy.data.cameras.new('MeasurementOrtho');cam=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(cam);scene.camera=cam;cd.type='ORTHO'
   def render(v):
    n=v['resolution'];span=v['span'];scene.render.resolution_x=scene.render.resolution_y=n;cd.ortho_scale=span;cd.clip_start=v['near'];cd.clip_end=v['far']
    camera=np.asarray(v['T_camera_from_semantic']);cam.matrix_world=Matrix((np.linalg.inv(camera)@np.diag([1,-1,-1,1])).tolist());bpy.context.view_layer.update()
    projection=np.asarray(cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),x=n,y=n,scale_x=1,scale_y=1))
    # Orthographic camera coefficient audit in the explicit pixel-edge convention.
    k=np.array([[projection[0,0]*n/2,0,(1+projection[0,3])*n/2],[0,projection[1,1]*n/2,(1-projection[1,3])*n/2],[0,0,1.]])
    assert np.max(np.abs(k-v['K_pixel_edges']))<.002
    path=out/f'_temporary_{n}_{v["name"]}.png';scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
    with Image.open(path) as image:alpha=np.array(image.getchannel('A'))
    path.unlink();return alpha
  else:
   import torch
   from gsplat.rendering import rasterization
   from scipy.spatial.transform import Rotation
   torch.set_num_threads(2);torch.empty(1,device='cuda')
   if fixture:
    xyz=np.array([[0.,0,0]]);q=np.array([[1.,0,0,0]]);scales=np.array([[.2,.1,.05]]);opacity=np.array([.9])
   else:
    from plyfile import PlyData
    data=PlyData.read(str(asset))['vertex'].data
    fields=lambda keys:np.stack([data[k] for k in keys],axis=-1).astype(float)
    xyz=fields(['x','y','z']);q=fields(['rot_0','rot_1','rot_2','rot_3']);scales=np.exp(fields(['scale_0','scale_1','scale_2']));opacity=torch.sigmoid(torch.as_tensor(data['opacity'].copy(),dtype=torch.float64)).numpy()
   assert all(np.isfinite(x).all() for x in [xyz,q,scales,opacity]);norm=np.linalg.norm(q,axis=-1,keepdims=True);assert (norm>0).all();q=q/norm
   rotations=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();basis=np.einsum('ij,njk->nik',axes,rotations)*scales[:,None,:];sigma=np.sqrt((basis*basis).sum(2));semantic=xyz@axes.T
   low=(semantic-6*sigma).min(0);high=(semantic+6*sigma).max(0);state['framing_rule']='All centres plus six-sigma semantic covariance support; framing only, not measured extent'
   tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32).contiguous();arrays=[tensor(v) for v in [xyz,q,scales,opacity,np.ones_like(xyz)]]
   state['color_note']='Constant colors used solely to request alpha; geometry/covariance/opacity unchanged; no appearance output claimed'
   def render(v):
    camera=np.asarray(v['T_camera_from_semantic']);A=np.eye(4);A[:3,:3]=axes;native=camera@A;n=v['resolution']
    with torch.inference_mode():
     _,alpha,_=rasterization(*arrays,tensor(native)[None],tensor(v['K_pixel_edges'])[None],n,n,near_plane=v['near'],far_plane=v['far'],packed=False,camera_model='ortho',rasterize_mode='classic',eps2d=.3,radius_clip=0.)
    assert torch.isfinite(alpha).all()
    return (alpha[0,:,:,0].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
  center=(low+high)/2;span=1.2*float((high-low).max());assert np.isfinite(span) and span>0
  state.update(framing_center=center.tolist(),framing_span=span,native_to_semantic=axes.tolist(),stage='render');save(out/'state.json',state)
  measured={};errors=[]
  for resolution in a.resolutions:
   views=[]
   for v in rig(center,span,resolution):
    alpha=render(v);path=out/f'alpha_{resolution}_{v["name"]}.png';Image.fromarray(alpha).save(path)
    row=dict(**v,path=str(path),sha256=sha(path))
    try:
     measurement=measure_view(alpha,v);row['measurement']=measurement;views.append(measurement)
    except ValueError as e:row['measurement_error']=str(e);errors.append(str(e))
    state['views'].append(row);save(out/'state.json',state)
   if len(views)==6:measured[str(resolution)]=combine(views)
  state.update(status='native_alpha_rendered',measurements=measured,measurement_errors=errors)
  if len(measured)==2:
   cv=convergence(measured[str(a.resolutions[0])],measured[str(a.resolutions[1])],span,a.resolutions[0]);state['convergence']=cv
   state['status']='measured_pending_cross_backend_review' if cv['passed'] else 'measurement_resolution_unstable'
   final=measured[str(a.resolutions[1])]
   if fixture:
    expected=np.array([2.,1,.5]) if mesh else 2*np.array([.2,.1,.05])*np.sqrt(-2*np.log(.5/.9))
    error=float(np.abs(np.array(final['extent'])-expected).max());state['fixture_check']=dict(expected_extent=expected.tolist(),max_extent_error=error,tolerance=2*span/a.resolutions[1],passed=error<=2*span/a.resolutions[1])
    if not state['fixture_check']['passed']:state['status']='fixture_extent_mismatch'
   else:
    ref=json.loads((R/'configs/reference_object_frames_v1/objects'/f'{a.sample_id}.json').read_text());state['candidate_calibration']=length_conditioned(axes,axes.T@np.asarray(final['center']),final['extent'][0],ref['size_length_width_height_m'][0])
  state['finished_at']=time.time();save(out/'state.json',state)
 except BaseException as error:
  state.update(status='failed',reason=repr(error),traceback=traceback.format_exc(),finished_at=time.time());save(out/'state.json',state);raise


if __name__=='__main__':main()
