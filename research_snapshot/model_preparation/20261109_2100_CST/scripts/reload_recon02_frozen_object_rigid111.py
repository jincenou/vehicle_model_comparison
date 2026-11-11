# Isolated rigid111 adapter; source SHA256: 097319bbe36f1993b72ed4758620c8f15b19ca23ec6bbe4bb9ea87b556e1d893
"""Independent native Gaussian reload with pixel comparison to generation previews."""
import os
os.environ['SPCONV_ALGO']='native'
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch

P=Path(__file__).resolve().parents[1]
S=P/'sources/recon02-f6720923260927333e22761fa722144d214733d9'
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True,type=Path)
args=parser.parse_args();O=args.run_dir.resolve()
assert O.is_relative_to((P/'runs/recon02_rigid111_v1').resolve())
os.chdir(S);sys.path.insert(0,str(S))
torch.empty(1,device='cuda')
from trellis.representations.gaussian import Gaussian
from trellis.utils import render_utils

state=json.loads((O/'state.json').read_text())
assert state['status']=='generated_pending_independent_reload'
for name,expected in state['output_hashes'].items():
    assert hashlib.sha256((O/name).read_bytes()).hexdigest()==expected
packed=torch.load(O/'native_gaussians.pt',map_location='cpu',weights_only=True)
assert packed['init_params']['sh_degree']==0
gs=Gaussian(**packed['init_params'])
for name,value in packed['params'].items():
    assert torch.isfinite(value).all(),name
    setattr(gs,name,value.cuda())
assert len(gs.get_xyz)==state['gaussians'] and len(gs.get_xyz)>0
extr,intr=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,.29670597],[.2,.2],2.,40.)
frames=render_utils.render_frames(gs,extr,intr,options={'resolution':512,'bg_color':(1,1,1)})
views=[]
for i,frame in enumerate(frames['color']):
    previous=np.array(Image.open(O/f'native_preview_{i}.png'))
    assert frame.shape==previous.shape==(512,512,3)
    diff=np.abs(frame.astype(np.int16)-previous.astype(np.int16))
    assert diff.max()<=1 and float(diff.mean())<=.01,(diff.max(),diff.mean())
    assert np.isfinite(frame).all()
    output=O/f'reload_gaussian_{i}.png';Image.fromarray(frame).save(output)
    views.append({'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                  'max_pixel_difference':int(diff.max()),'mean_pixel_difference':float(diff.mean()),'foreground_pixels_rgb_lt250':int(np.count_nonzero(frame.min(axis=2)<250)),'sparse_or_white_diagnostic':bool(np.count_nonzero(frame.min(axis=2)<250)<=100)})
result={'status':'render_passed_pending_process_exit_and_visual','gaussians':len(gs.get_xyz),
        'checkpoint_sha256':state['output_hashes']['native_gaussians.pt'],'views':views,
        'scope':'Native independent reload only; not common-camera benchmark metrics'}
(O/'reload_validation.json').write_text(json.dumps(result,indent=2)+'\n')
