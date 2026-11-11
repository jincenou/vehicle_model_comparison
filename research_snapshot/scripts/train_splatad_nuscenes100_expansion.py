"""Native SplatAD trainer, fixed configuration and explicit short health gate."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'adapters'))
import numpy as np
import torch
from PIL import Image
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import Trainer
from nerfstudio.scripts.train import _set_random_seed
from nuscenes_train_only import TrainOnlyNuScenesConfig

def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.json');t.write_text(json.dumps(d,indent=2));t.replace(p)

class MonitoredTrainer(Trainer):
    def save_checkpoint(self,step):
        super().save_checkpoint(step)
        if step!=2000 or self.config.max_num_iterations!=30001:return
        path=self.checkpoint_dir/f'step-{step:09d}.ckpt'
        state=torch.load(path,map_location='cpu')
        assert state['step']==step
        for key,value in state['pipeline'].items():
            if torch.is_tensor(value) and value.is_floating_point():
                assert torch.isfinite(value).all(),f'Nonfinite checkpoint tensor: {key}'
        del state
        self.pipeline.eval()
        previews={}
        for name,dataset in [('source',self.pipeline.datamanager.train_dataset),('train_shadow',self.pipeline.datamanager.eval_dataset)]:
            camera=dataset.cameras[0:1].to(self.device);camera.metadata['cam_idx']=0
            with torch.no_grad():rgb=self.pipeline.model.get_outputs_for_camera(camera=camera)['rgb'].detach().cpu()
            assert torch.isfinite(rgb).all() and rgb.std()>1e-5
            out=self.base_dir/f'checkpoint2000_{name}.png'
            Image.fromarray((rgb.clamp(0,1).numpy()*255).astype(np.uint8)).save(out)
            previews[name]=str(out)
        self.pipeline.train()
        write(self.base_dir/'checkpoint2000_validation.json',{'status':'passed','step':step,'checkpoint':str(path),
              'tensor_reload':'all floating checkpoint tensors finite','native_renders':previews,'time':time.time()})

    def train_iteration(self,step):
        result=super().train_iteration(step)
        loss=result[0]
        assert torch.isfinite(loss).all(),f'Nonfinite training loss at step {step}'
        if step%25==0:
            item={'status':'training','scene':self.config.experiment_name,'step':step,'time':time.time(),
                  'loss':float(loss.detach()),'gaussians':int(self.pipeline.model.means.shape[0]),
                  'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20,'pid':os.getpid()}
            write(self.base_dir/'health.json',item)
            print('HEALTH '+json.dumps(item),flush=True)
        return result

def validate_native(trainer,scene,stage,scope):
    model=trainer.pipeline.model;manager=trainer.pipeline.datamanager
    trainer.pipeline.eval()
    camera=manager.train_dataset.cameras[0:1].to(trainer.device)
    camera.metadata['cam_idx']=0
    with torch.no_grad():before=model.get_outputs_for_camera(camera=copy.deepcopy(camera))['rgb'].detach().cpu()
    assert torch.isfinite(before).all() and before.std()>1e-5
    ckpt=sorted(trainer.checkpoint_dir.glob('step-*.ckpt'))[-1]
    saved=torch.load(ckpt,map_location='cpu')
    assert saved['step']==(100 if stage=='pilot' else 30000),saved['step']
    trainer.pipeline.load_pipeline(saved['pipeline'],saved['step'],strict=True)
    trainer.pipeline.eval()
    with torch.no_grad():after=model.get_outputs_for_camera(camera=copy.deepcopy(camera))['rgb'].detach().cpu()
    error=float((before-after).abs().max())
    assert error<1e-5,error
    tag={'human':'splatad_human_reviewed95',
         'vehicle':'splatad_reviewed288_train_only',
         'reconstruction_only':'splatad_nuscenes100_expansion'}[scope]
    out=R/'exports'/tag/stage/scene;out.mkdir(parents=True,exist_ok=True)
    Image.fromarray((after.clamp(0,1).numpy()*255).astype(np.uint8)).save(out/'native_reload_preview.png')
    ids=model.id.detach().cpu().reshape(-1)
    actors=manager.train_dataparser_outputs.metadata['trajectories']
    bundles=[]
    for i,actor in enumerate(actors):
        mask=ids==i
        if not mask.any():continue
        # Native appearance features and all decoder/trajectory parameters are retained.
        # No pseudo-RGB PLY is used for visual metrics.
        bundle={'scene':scene,'actor_index':i,'uuid':actor['uuid'],'label':actor['label'],
                'gauss_params':{k:v.detach().cpu()[mask] for k,v in model.gauss_params.items()},
                'trajectory':actor,'native_checkpoint':str(ckpt),
                'appearance_contract':'Native SplatAD RGB decoder + scene appearance embedding required; features are not SH colours.'}
        path=out/(actor['uuid']+'.pt');torch.save(bundle,path)
        check=torch.load(path,map_location='cpu')
        assert check['gauss_params']['means'].shape[0]==int(mask.sum())
        bundles.append({'uuid':actor['uuid'],'gaussians':int(mask.sum()),'bundle':str(path)})
    result={'status':'passed','scene':scene,'stage':stage,'checkpoint':str(ckpt),'step':saved['step'],
            'reload_rgb_max_abs':error,'preview':str(out/'native_reload_preview.png'),'actors':bundles,
            'scene_gaussians':len(ids),'split_sha256':trainer.config.pipeline.datamanager.dataparser.split_sha256,
            'real_heldout_eligible':scope!='reconstruction_only',
            'heldout_reason':'no_independent_real_heldout' if scope=='reconstruction_only' else None,
            'static_vehicle_policy':'Native parser assigns stationary targets to background. Separate target extraction validation remains required.',
            'time':time.time()}
    write(R/'reports'/f'{tag}_{stage}_{scene}_validation.json',result)
    print('VALIDATION '+json.dumps({k:v for k,v in result.items() if k!='actors'}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--stage',choices=['pilot','formal'],required=True);p.add_argument('--scope',choices=['human','vehicle','reconstruction_only'],default='human');p.add_argument('--attempt');p.add_argument('--resume',action='store_true');a=p.parse_args()
    preflight={'human':'nuscenes_human_reviewed95_preflight.json',
               'vehicle':'nuscenes_reviewed288_preflight.json',
               'reconstruction_only':'nuscenes100_expansion_preflight.json'}[a.scope]
    pre=json.loads((R/'reports'/preflight).read_text())
    row=next(x for x in pre['scenes'] if x['scene']==a.scene)
    assert row['status']==('ready_reconstruction_only' if a.scope=='reconstruction_only' else 'ready')
    isolate_tag={'human':'human_reviewed95','vehicle':'reviewed288',
                 'reconstruction_only':'nuscenes100_expansion'}[a.scope]
    isolation=json.loads((R/'reports/splatad_train_only'/isolate_tag/f'{a.scene}.json').read_text())
    assert isolation['status']==('verified_reconstruction_only' if a.scope=='reconstruction_only' else 'verified_train_only')
    assert isolation['source_split_sha256']==row['manifest_sha256' if a.scope=='reconstruction_only' else 'split_sha256']
    config=copy.deepcopy(method_configs['splatad'])
    config._target=MonitoredTrainer
    run_tag={'human':'splatad_human_reviewed95','vehicle':'splatad_reviewed288_train_only',
             'reconstruction_only':'splatad_nuscenes100_expansion'}[a.scope]
    attempt=a.attempt or {'human':'human_reviewed95_train_only_v1',
                          'vehicle':'reviewed288_train_only_v2',
                          'reconstruction_only':'nuscenes100_expansion_v1'}[a.scope]
    config.output_dir=R/'runs'/run_tag/a.stage;config.experiment_name=a.scene;config.timestamp=attempt
    config.vis='tensorboard';config.viewer.quit_on_train_completion=True
    config.machine.num_devices=1;config.machine.seed=42
    config.pipeline.datamanager.dataparser=TrainOnlyNuScenesConfig(data=Path(isolation['data_root']),sequence=a.scene,split_manifest=Path(isolation['manifest']),split_sha256=isolation['manifest_sha256'])
    # Explicit storage placement, no changes to image resolution or native learning schedule.
    config.pipeline.datamanager.cache_images='cpu';config.pipeline.datamanager.cache_lidars='cpu'
    config.pipeline.datamanager.max_thread_workers=4
    config.early_stopping_tracker.metric=None;config.checkpoint_saving_tracker.metric=None
    config.steps_per_eval_batch=0;config.steps_per_eval_image=0;config.steps_per_eval_all_images=0
    # Heldout evaluation is a separate post-training job; it never selects a checkpoint.
    final_step=100 if a.stage=='pilot' else 30000
    config.max_num_iterations=final_step+1
    config.steps_per_save=100 if a.stage=='pilot' else 2000
    config.logging.steps_per_log=100
    base=config.get_base_dir()
    if a.resume:
        assert (base/'config.yml').exists(),f'Cannot resume missing run {base}'
        checkpoints=sorted(config.get_checkpoint_dir().glob('step-*.ckpt'))
        assert checkpoints,f'Cannot resume without checkpoint {base}'
        checkpoint=checkpoints[-1]
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        start_step=int(saved['step'])+1
        assert 0<start_step<=final_step,f'Unexpected checkpoint step {start_step}'
        config.load_checkpoint=checkpoint
        config.max_num_iterations=final_step+1-start_step
        print('RESUME '+json.dumps({'scene':a.scene,'checkpoint':str(checkpoint),'start_step':start_step,'final_step':final_step}),flush=True)
        del saved
    else:
        assert not (base/'config.yml').exists(),f'Refuse overwriting existing run {base}'
        config.save_config()
    _set_random_seed(42)
    # Fail explicitly on missing weights; no implicit downloads into an old cache.
    def forbidden_download(*args,**kwargs):raise RuntimeError('Unplanned runtime weight download; prepare exact dependency first')
    torch.hub.download_url_to_file=forbidden_download
    trainer=config.setup(local_rank=0,world_size=1);trainer.setup();trainer.train()
    validate_native(trainer,a.scene,a.stage,a.scope)
if __name__=='__main__':main()
