"""Re-render unchanged GLBs with normalized heading, shared view settings and light rig."""
import argparse,json,hashlib,math
from pathlib import Path
import bpy
from mathutils import Vector,Matrix
p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];P=R/'model_preparation/20261109_2100_CST';O=R/'reports/visual_aligned_20261110'
for sid in ['scene-0064_t000961','scene-0063_t000792']:
 out=O/sid/a.method;out.mkdir(parents=True,exist_ok=False)
 folder=P/f'runs/{a.method}_vehicle432_v1/{sid}';state=json.loads((folder/'state.json').read_text());path=Path(state['output']);assert hashlib.sha256(path.read_bytes()).hexdigest()==state['output_sha256']
 bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(path));meshes=[x for x in bpy.context.scene.objects if x.type=='MESH'];assert meshes
 # Front direction checked from real input and existing front/rear/side previews.
 # TRELLIS.2's GLB z-up/y-up conversion is inverted by Blender's glTF importer.
 front=Vector((1,0,0))if a.method=='hunyuan21'and sid=='scene-0063_t000792'else Vector((0,-1,0))
 up=Vector((0,0,1));left=up.cross(front);axes=Matrix((front,left,up));A=axes.to_4x4()
 for obj in meshes:obj.matrix_world=A@obj.matrix_world
 corners=[obj.matrix_world@Vector(c)for obj in meshes for c in obj.bound_box];lo=Vector([min(v[i]for v in corners)for i in range(3)]);hi=Vector([max(v[i]for v in corners)for i in range(3)]);center=(lo+hi)/2;L=hi.x-lo.x;assert L>0
 scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=42;scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.film_transparent=True;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard'
 world=bpy.data.worlds.new('SharedNeutral');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.8,.8,.8,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6;scene.world=world
 for i,offset in enumerate([(2,-3,4),(-3,1,2)]):
  ld=bpy.data.lights.new('Area'+str(i),'AREA');ld.energy=350*L**2;ld.shape='DISK';ld.size=2*L;light=bpy.data.objects.new(ld.name,ld);scene.collection.objects.link(light);light.location=center+Vector(offset)*L;light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
 cd=bpy.data.cameras.new('SharedCamera');cam=bpy.data.objects.new(cd.name,cd);scene.collection.objects.link(cam);cd.type='PERSP';cd.sensor_fit='HORIZONTAL';cd.angle=math.radians(40);cd.clip_start=.01*L;cd.clip_end=20*L;scene.camera=cam;views=[]
 for name,az in [('front_45',45),('side_90',90),('rear_135',135)]:
  az,el=math.radians(az),math.radians(12);cam.location=center+Vector((math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)))*1.8*L;cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True);views.append(dict(name=name,camera_location=list(cam.location),camera_rotation=list(cam.rotation_euler)))
 (out/'view_manifest.json').write_text(json.dumps(dict(sample_id=sid,method=a.method,asset=str(path),asset_sha256=state['output_sha256'],blender_front_before_alignment=list(front),native_to_semantic=[list(v)for v in axes],front='+X',left='+Y',up='+Z',centering='transformed mesh bounding box',length=L,camera_distance_over_length=1.8,vertical_fov_degrees=40,elevation_degrees=12,views=views,scope='heading-normalized qualitative display; NOT physical registration or metric camera',renderer='Blender Cycles CPU16, shared neutral lighting'),indent=2)+'\n')
 print(sid,a.method,'done',flush=True)
