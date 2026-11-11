"""Development render using native alpha extent and explicit pixel coordinates.

Candidate measurement only: cross-backend/physical qualification remains pending.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

R=Path(__file__).resolve().parents[1]
MODULES=R/'evaluation_nuscenes_v1/scripts'
if not MODULES.exists():MODULES=R.parent/'evaluation'
sys.path.insert(0,str(MODULES))
from reference_camera_rig import check_pose, project
from pixel_coordinates import intrinsics_for_convention
from spatial_calibration import length_conditioned

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample-id',required=True);p.add_argument('--method',required=True)
    a=p.parse_args()
    fp=R/'configs/reference_object_frames_v1/objects'/f'{a.sample_id}.json'
    ref=json.loads(fp.read_text())
    dp=R/'exports/asset_orientation_final_v1'/a.method/a.sample_id/'orientation_render.json'
    display=json.loads(dp.read_text());asset=Path(display['asset'])
    assert sha(asset)==display['asset_sha256']
    heading=Path(display['heading_decision']);assert sha(heading)==display['heading_decision_sha256']
    decision=json.loads(heading.read_text());assert decision['asset_sha256']==sha(asset)
    measurement_path=R/'exports/calibration_alpha_pilot_v1'/a.sample_id/a.method/'state.json'
    measurement=json.loads(measurement_path.read_text())
    assert measurement['status']=='measured_pending_cross_backend_review'
    assert measurement['asset_sha256']==display['asset_sha256'] and measurement['heading_sha256']==sha(heading)
    assert measurement['sample_id']==a.sample_id and measurement['method']==a.method
    assert measurement['convergence']['passed']
    for view in measurement['views']:assert sha(view['path'])==view['sha256']
    axes=np.asarray(measurement['native_to_semantic'])
    assert np.allclose(axes,decision['native_to_semantic'],rtol=0,atol=1e-10)
    dimensions=measurement['measurements'][str(measurement['resolutions'][-1])]
    contract=length_conditioned(axes,axes.T@np.asarray(dimensions['center']),dimensions['extent'][0],ref['size_length_width_height_m'][0])
    A=np.asarray(contract['T_reference_from_input']);scale=contract['scale_reference_units_per_input_unit']
    out=R/'exports/asset_registration_pilot_v2'/a.sample_id/a.method
    out.mkdir(parents=True,exist_ok=False)
    cameras=[dict(name='source',K=ref['source_K'],wh=ref['source_render_wh'],
                  T_camera_from_object=ref['T_source_camera_from_object'],near=.01,far=200)]
    for v in ref['orbit']['views']:
        if v['index'] in [0,4,8]:
            cameras.append(dict(name=f'orbit_{v["index"]:02d}',K=ref['orbit']['K'],wh=ref['orbit']['render_wh'],
                                T_camera_from_object=v['T_camera_from_object'],near=v['near_m'],far=v['far_m']))
    records=[]
    if asset.suffix=='.ply':
        import torch
        from gsplat.rendering import rasterization
        from PIL import Image
        torch.set_num_threads(4)
        from plyfile import PlyData
        data=PlyData.read(str(asset))['vertex'].data
        v={name:data[name].copy() for name in data.dtype.names}
        assert not any(k.startswith('f_rest_')for k in v), 'Pilot adapter is SH0-only; do not discard higher-order SH'
        fields=lambda keys:np.stack([v[k] for k in keys],axis=-1)
        xyz=fields(['x','y','z']);q=fields(['rot_0','rot_1','rot_2','rot_3']);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        scales=np.exp(fields(['scale_0','scale_1','scale_2']));opacity=1/(1+np.exp(-v['opacity']))
        rgb=np.clip(fields(['f_dc_0','f_dc_1','f_dc_2'])*.28209479177387814+.5,0,1)
        assert all(np.isfinite(x).all()for x in [xyz,q,scales,opacity,rgb])
        tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32).contiguous()
        arrays=[tensor(x)for x in [xyz,q,scales,opacity,rgb]]
        # Keep Gaussians, anisotropic covariance and their coordinate convention intact.
        # Similarity scale is factored out of the camera; projection is unchanged.
        for cam in cameras:
            V=np.array(cam['T_camera_from_object'])@A;V[:3,:]/=scale;check_pose(V)
            K=np.array(cam['K']);w,h=cam['wh']
            raster_k=intrinsics_for_convention(K,'integer_centres','pixel_edges')
            with torch.inference_mode():
                im,alpha,_=rasterization(*arrays,tensor(V)[None],tensor(raster_k)[None],w,h,
                    backgrounds=torch.ones((1,3),device='cuda'),near_plane=cam['near']/scale,
                    far_plane=cam['far']/scale,packed=False)
            assert torch.isfinite(im).all()and torch.isfinite(alpha).all()
            color=(im[0,:,:,:3].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
            mask=(alpha[0,:,:,0].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
            Image.fromarray(color).save(out/(cam['name']+'.png'));Image.fromarray(mask).save(out/(cam['name']+'_alpha.png'))
            probe=xyz[::max(1,len(xyz)//1000)].astype(float)
            obj=probe@A[:3,:3].T+A[:3,3]
            x,z=project(probe,V,K);y,_=project(obj,cam['T_camera_from_object'],K)
            visible=z>.01/scale;err=float(np.max(np.abs(x[visible]-y[visible])))if visible.any()else 0
            assert err<1e-6
            records.append(dict(**cam,render_camera_in_native_units=V.tolist(),scale_projection_max_error_px=err,
                                renderer_K_pixel_edges=raster_k.tolist(),reference_pixel_convention='integer_centres',
                                alpha_max=float(alpha.max()),foreground_pixels=int((mask>127).sum())))
        renderer='gsplat original SH0, covariance preserved by inverse camera transform'
    else:
        import bpy
        from mathutils import Matrix,Vector
        bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(asset))
        meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert meshes
        # Source coordinates of the accepted heading are Blender-import world Z-up.
        bpy.context.view_layer.update()
        worlds={obj.name:obj.matrix_world.copy() for obj in meshes}
        def depth(obj):
            count=0
            while obj.parent is not None:count+=1;obj=obj.parent
            return count
        for obj in sorted(meshes,key=depth):
            obj.matrix_world=Matrix(A.tolist())@worlds[obj.name]
            bpy.context.view_layer.update()
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=42
        scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_percentage=100
        scene.render.film_transparent=True;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
        scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
        world=bpy.data.worlds.new('Neutral');world.use_nodes=True
        world.node_tree.nodes['Background'].inputs['Color'].default_value=(.8,.8,.8,1)
        world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6;scene.world=world
        length=ref['size_length_width_height_m'][0]
        for i,offset in enumerate([(2,-3,4),(-3,1,2)]):
            ld=bpy.data.lights.new('Area'+str(i),'AREA');ld.energy=350*length**2;ld.shape='DISK';ld.size=2*length
            light=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(light)
            light.location=Vector(offset)*length;light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
        cd=bpy.data.cameras.new('FixedReference');camera=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(camera);scene.camera=camera
        cd.type='PERSP';cd.sensor_fit='HORIZONTAL';cd.sensor_width=36
        from PIL import Image
        for cam in cameras:
            K=np.asarray(cam['K']);w,h=cam['wh'];Tco=np.asarray(cam['T_camera_from_object'])
            scene.render.resolution_x=w;scene.render.resolution_y=h
            ratio=K[0,0]/K[1,1];scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=ratio
            cd.lens=K[0,0]*36/w;cd.shift_x=((w-1)/2-K[0,2])/w;cd.shift_y=(K[1,2]-(h-1)/2)*ratio/w
            cd.clip_start=cam['near'];cd.clip_end=cam['far']
            camera.matrix_world=Matrix((np.linalg.inv(Tco)@np.diag([1,-1,-1,1])).tolist())
            bpy.context.view_layer.update()
            P=np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),x=w,y=h,scale_x=1,scale_y=ratio))
            actual=np.array([[P[0,0]*w/2,0,(1-P[0,2])*w/2-.5],[0,P[1,1]*h/2,(1+P[1,2])*h/2-.5],[0,0,1]])
            err=float(np.max(np.abs(actual-K)));assert err<.001, f'Blender intrinsics mismatch {err}'
            rgba=out/(cam['name']+'_rgba.png');scene.render.filepath=str(rgba);bpy.ops.render.render(write_still=True)
            tile=Image.open(rgba).convert('RGBA');arr=np.array(tile.getchannel('A'))
            white=Image.new('RGBA',tile.size,'white');white.alpha_composite(tile);white.convert('RGB').save(out/(cam['name']+'.png'))
            tile.getchannel('A').save(out/(cam['name']+'_alpha.png'))
            records.append(dict(**cam,blender_calculated_K=actual.tolist(),intrinsics_max_error_px=err,
                                alpha_max=float(arr.max()/255),foreground_pixels=int((arr>127).sum())))
        renderer='Blender Cycles CPU, native GLB materials, fixed shared reference-length lighting'
    for record in records:
        record['rgb']=str(out/(record['name']+'.png'));record['rgb_sha256']=sha(record['rgb'])
        record['alpha']=str(out/(record['name']+'_alpha.png'));record['alpha_sha256']=sha(record['alpha'])
    result=dict(status='development_registered_render',sample_id=a.sample_id,method=a.method,
                reference_contract=str(fp),reference_contract_sha256=sha(fp),heading_decision=str(heading),heading_sha256=sha(heading),
                display_contract=str(dp),display_contract_sha256=sha(dp),asset=str(asset),asset_sha256=sha(asset),
                adapter_sha256=sha(__file__),T_reference_object_from_input=A.tolist(),scale_m_per_input_unit=float(scale),
                measurement_contract=str(measurement_path),measurement_contract_sha256=sha(measurement_path),
                pixel_coordinates_adapter_sha256=sha(MODULES/'pixel_coordinates.py'),
                registration_rule='accepted heading + six-view alpha-support centre and extent + isotropic annotated-length conditioning',
                registration_limit='Native alpha measurement is a candidate only; NOT approved for formal scoring',
                pixel_convention_note='Integer-centre reference K; explicit +0.5 conversion for gsplat; Blender sensor shifts use integer-centre K. Raw source convention still requires native-render validation.',
                renderer=renderer,views=records,formal_metrics_qualified=False,source_reference_eligible=False,
                reason='Development object has source-frame truncation; physical registration and lighting require validation.',
                raw_asset_modified=False,metric_score_fitting=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(method=a.method,status=result['status'],output=str(out))))

if __name__=='__main__':main()
