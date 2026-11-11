"""Reload the exported PBR GLB in a separate Blender process and render it."""
import hashlib
import json
import math
import argparse
from pathlib import Path
import bpy
from mathutils import Vector

P = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=Path, default=P/'runs/hunyuan21_official_demo_v1')
args = parser.parse_args()
O = args.run_dir.resolve()
assert O.is_relative_to((P/'runs').resolve())
state = json.loads((O/'state.json').read_text())
path = Path(state['output'])
assert hashlib.sha256(path.read_bytes()).hexdigest() == state['output_sha256']
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(path))
meshes = [x for x in bpy.context.scene.objects if x.type == 'MESH']
assert meshes
corners = [obj.matrix_world @ Vector(c) for obj in meshes for c in obj.bound_box]
low = Vector([min(x[i] for x in corners) for i in range(3)])
high = Vector([max(x[i] for x in corners) for i in range(3)])
center = (low+high)/2
extent = max(high-low)
assert extent > 0
images = [image for image in bpy.data.images if image.size[0] > 0]
assert images, 'Exported PBR textures were not reloaded'
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 16
scene.render.threads_mode = 'FIXED'
scene.render.threads = 4
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('NeutralWorld')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (.8,.8,.8,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = .6
scene.world = world
for index,offset in enumerate([(2,-3,4),(-3,1,2)]):
    light_data = bpy.data.lights.new('Area'+str(index),'AREA')
    light_data.energy = 350*extent**2
    light_data.shape = 'DISK';light_data.size=extent*2
    light = bpy.data.objects.new(light_data.name,light_data);scene.collection.objects.link(light)
    light.location=center+Vector(offset)*extent
    light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
camera_data=bpy.data.cameras.new('ValidationCamera')
camera=bpy.data.objects.new('ValidationCamera',camera_data);scene.collection.objects.link(camera)
camera_data.lens=50;scene.camera=camera
views=[]
for angle in [0,17]:
    az=math.radians(angle)
    camera.location=center+Vector((3*math.sin(az),-3*math.cos(az),1.2))*extent
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(O/f'reload_pbr_{angle:02d}.png')
    bpy.ops.render.render(write_still=True)
    views.append(scene.render.filepath)
result={'status':'rendered_pending_visual_and_process_exit','output_sha256':state['output_sha256'],
        'mesh_objects':len(meshes),'texture_images':len(images),'views':views,
        'renderer':'Blender 4.0 Cycles CPU, 16 samples, Standard, transparent film',
        'scope':'native output reload only; not common-camera benchmark metrics',
        'sample_id':state.get('sample_id'), 'run_dir':str(O)}
(O/'reload_render_validation.json').write_text(json.dumps(result,indent=2)+'\n')
