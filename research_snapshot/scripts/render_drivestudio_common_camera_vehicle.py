"""Render a native DriveStudio actor at the frozen raw-source camera."""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
torch.empty(1,device="cuda")  # Claim the assigned idle GPU before loading the dataset.
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(('/opt/rigid-benchmark/drivestudio'.replace('/opt/rigid-benchmark/drivestudio', __import__("os").environ.get('RIGID_DRIVESTUDIO', '/opt/rigid-benchmark/drivestudio')).replace('/opt/rigid-benchmark/external', __import__("os").environ.get('RIGID_EXTERNAL_ROOT', '/opt/rigid-benchmark/external'))))
sys.path.insert(0,str(REPO))
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.gaussians.basics import dataclass_gs
from datasets.base.pixel_source import get_rays
import common_camera_contract
from common_camera_contract import load_contract, projection_audit

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def image(t,p):
    a=t.detach().cpu().numpy()
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray((np.clip(a,0,1)*255).round().astype('uint8')).save(p)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample-id');g.add_argument('--human-id');parser.add_argument('--method',choices=['omnire','streetgs'],required=True);a=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    inventory=json.loads((ROOT/'configs/common_camera_vehicle432_v2/manifest.json').read_text())
    entry=next(x for x in inventory['objects']if x['sample_id']==a.sample_id)
    selected=entry['object']
    common_camera_contract.CONTRACT=Path(entry['contract'])
    assert sha(common_camera_contract.CONTRACT)==entry['contract_sha256']
    assert a.sample_id==selected['sample_id']
    scene=selected['scene'];uuid=selected['native_instance_token'];source=Path(selected['anchor']['source_rgb']).resolve()
    assert sha(source)==selected['anchor']['source_rgb_sha256']
    item={'native_id':'nuscenes:'+uuid,'raw_timestamp_us':selected['camera']['timestamp_us']}
    node_type='RigidNodes'
    assert entry['methods'][a.method]['status']=='native_actor_candidate_pending_source_render'
    state=entry['methods'][a.method]['training_job'];assert state['status']=='formal_validated'
    checkpoint=Path(state['checkpoint']);assert sha(checkpoint)==state['checkpoint_sha256']
    cfg=OmegaConf.load(checkpoint.parent/'config.yaml');data=Path(cfg.data.data_root)/'000'
    out=ROOT/'exports/common_camera_vehicle432_v1'/a.method/a.sample_id;out.mkdir(parents=True,exist_ok=False)
    def unavailable(reason):
        (out/'validation.json').write_text(json.dumps({'status':'unavailable','sample_id':a.sample_id,'scene':scene,'reason':reason,'uuid':uuid,'checkpoint':str(checkpoint)},indent=2)+'\n')
    infos=json.loads((data/'instances/instances_info.json').read_text())
    true_ids=[int(k) for k,x in infos.items() if x['id']==uuid]
    if not true_ids:
        unavailable('native_identity_absent_from_training_instances');return
    assert len(true_ids)==1, 'Native identity ambiguous'
    source_sha=sha(source)
    source_matches=[p for p in (data/'images').glob('*.jpg') if p.stat().st_size==source.stat().st_size and sha(p)==source_sha]
    if not source_matches:
        unavailable('exact_frozen_source_image_absent_from_training_input');return
    assert len(source_matches)==1, 'Exact source image ambiguous in native input'
    frame,cam=map(int,source_matches[0].stem.split('_'))
    dataset=DrivingDataset(data_cfg=cfg.data)
    init_cfg=OmegaConf.to_container(cfg.model[node_type].init,resolve=True)
    native=dataset.get_init_objects(cur_node_type=node_type,**init_cfg)
    native_keys=list(native)
    dataset_ids=[i for i in native_keys if int(dataset.pixel_source.instances_true_id[i])==true_ids[0]]
    if not dataset_ids:
        unavailable('native_actor_filtered_by_original_init')
        return
    assert len(dataset_ids)==1
    instance_index=native_keys.index(dataset_ids[0])
    trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2,3),device=torch.device('cuda'))
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint),load_only_model=True)
    trainer.set_eval();model=trainer.models[node_type]
    assert model.num_instances==len(native_keys)
    # Native insertion order is reproduced from the exact training data/config.
    # Sizes and visibility are unchanged by optimization and verify the ID map.
    for index,key in enumerate(native_keys):
        assert torch.allclose(model.instances_size[index].cpu(),native[key]['size'].cpu(),atol=1e-5)
        assert torch.equal(model.instances_fv[:,index].cpu(),native[key]['frame_info'].cpu())
    del native
    unique_cam=dataset.pixel_source.camera_data[cam].unique_cam_idx
    idx=frame*dataset.pixel_source.num_cams+unique_cam
    image_info,camera_info=dataset.pixel_source.get_image(idx)
    image_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in image_info.items()}
    camera_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in camera_info.items()}
    contract=load_contract(a.sample_id)
    world_from_model=np.loadtxt(data/'extrinsics'/f'{cfg.data.start_timestep:03d}_0.txt')
    model_from_world=np.linalg.inv(world_from_model)
    model_from_camera=model_from_world @ np.asarray(contract['T_world_from_camera_opencv'])
    assert np.max(np.abs(camera_info['camera_to_world'].cpu().numpy()-model_from_camera))<1e-4
    width,height=contract['render_wh']
    camera_info['camera_to_world']=torch.tensor(model_from_camera,dtype=torch.float32,device='cuda')
    camera_info['intrinsics']=torch.tensor(contract['K_raw'],dtype=torch.float32,device='cuda')
    camera_info['width']=torch.tensor(width,device='cuda')
    camera_info['height']=torch.tensor(height,device='cuda')
    x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
    origins,viewdirs,direction_norm=get_rays(x.flatten(),y.flatten(),camera_info['camera_to_world'],camera_info['intrinsics'])
    image_info={k:image_info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
    image_info.update(origins=origins.reshape(height,width,3),viewdirs=viewdirs.reshape(height,width,3),
        direction_norm=direction_norm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
    # novel_view disables camera refinement only; the native source timestamp and appearance ID are retained.
    full=trainer(image_info,camera_info,novel_view=True)
    camera=trainer.process_camera(camera_infos=camera_info,image_ids=image_info['img_idx'].flatten()[0],novel_view=True)
    camera_audit=projection_audit(contract,model_from_world,camera.camtoworlds.cpu().numpy(),camera.Ks.cpu().numpy())
    assert bool(model.instances_fv[model.cur_frame,instance_index])
    gs=model.get_gaussians(camera);mask=model.point_ids.reshape(-1)==instance_index
    if int(mask.sum())==0:
        unavailable('native_actor_has_zero_gaussians_after_training');return
    subset={k:v[mask] for k,v in gs.items()}
    actor=dataclass_gs(**subset,detach_keys=[],extras=None)
    render,_=trainer.render_gaussians(gs=actor,cam=camera,near_plane=trainer.render_cfg.near_plane,
        far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
    alpha=render['opacity'];raw=render['rgb_gaussians']
    # Apply the native affine to foreground color; the affine bias is alpha weighted.
    bias=trainer.affine_transformation(torch.zeros_like(raw),image_info)
    rgb=(trainer.affine_transformation(raw,image_info)-bias*(1-alpha)+(1-alpha)).clamp(0,1)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert float(alpha.max())>0 and float(rgb.std())>1e-5
    image(full['rgb'],out/'native_full_scene_source.png');image(rgb,out/'native_actor_source.png');image(alpha,out/'native_actor_alpha.png')
    bundle={'sample_id':a.sample_id,'uuid':uuid,'native_instance_index':instance_index,
        'native_dataset_id':dataset_ids[0],'source_frame':frame,'native_cur_frame':int(model.cur_frame),
        'checkpoint':str(checkpoint),'posed_gaussians':{k:v.cpu() for k,v in subset.items()},
        'canonical_parameters':{k:v[mask].detach().cpu() for k,v in model.state_dict().items()
                                if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==mask.numel()},
        'camera_K':camera.Ks.cpu(),'camera_to_world':camera.camtoworlds.cpu(),
        'appearance_contract':'Posed RGB is source-view SH evaluation; retain checkpoint/SH and native renderer for other views.'}
    torch.save(bundle,out/'native_actor_snapshot.pt')
    result={'status':'passed','sample_id':a.sample_id,'scene':scene,'method':a.method,
        'source_image':str(source),'source_image_sha256':source_sha,'processed_source_image':str(source_matches[0]),
        'native_id':item['native_id'],'actor_uuid':uuid,'actor_index':instance_index,'gaussians':int(mask.sum()),
        'source_frame':frame,'source_camera_index':int(idx),'native_cur_frame':int(model.cur_frame),
        'native_node_type':node_type,'frozen_raw_timestamp_us':item.get('raw_timestamp_us'),
        'checkpoint':str(checkpoint),'step':int(trainer.step),'bundle':str(out/'native_actor_snapshot.pt'),
        'real_heldout_eligible':False,
        'identity_validation':'reproduced native initialization order; all instance sizes and frame visibility exactly checked',
        'renderer':f'native DriveStudio {a.method} renderer, SH and alpha-weighted native affine, white background',
        'common_camera_audit':camera_audit,
        'actor_preview':str(out/'native_actor_source.png'),'actor_alpha':str(out/'native_actor_alpha.png'),
        'full_scene_preview':str(out/'native_full_scene_source.png'),'render_wh':[int(camera.W),int(camera.H)],
        'camera_K':camera.Ks.cpu().tolist(),'camera_to_world':camera.camtoworlds.cpu().tolist()}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
