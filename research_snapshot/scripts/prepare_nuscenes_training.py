"""Freeze native sensor streams and source-protected time-block splits, without changing raw data."""
import collections, hashlib, json, time
from pathlib import Path

R=Path(__file__).resolve().parents[1]
RAW=Path(('/data/nuScenes'.replace('/data', __import__("os").environ.get('RIGID_DATA_ROOT', '/data')))); VERSION='v1.0-trainval'
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def load(name):
    print('reading',name,flush=True)
    return json.loads((RAW/VERSION/(name+'.json')).read_text())
def strings(value):
    if isinstance(value,dict):
        for v in value.values(): yield from strings(v)
    elif isinstance(value,list):
        for v in value: yield from strings(v)
    elif isinstance(value,str): yield value
def main():
    selection=json.loads((R/'inputs/vehicle53_selection.json').read_text())
    rows=[s for s in selection['samples'] if s['dataset']=='nuscenes']
    names=sorted({s['scene_id'] for s in rows})
    scenes=[s for s in load('scene') if s['name'] in names]
    scene_tokens={s['token']:s['name'] for s in scenes}
    samples=[s for s in load('sample') if s['scene_token'] in scene_tokens]
    sample_scene={s['token']:scene_tokens[s['scene_token']] for s in samples}
    # All nested production provenance is conservatively protected, including FLUX anchors,
    # merged views and candidate observations; never choose heldout based on model scores.
    protected={name:set() for name in names}
    for row in rows:
        for value in strings(row):
            if value.startswith(str(RAW)+'/samples/') or value.startswith(str(RAW)+'/sweeps/'):
                protected[row['scene_id']].add(str(Path(value).relative_to(RAW)))
    sd=[s for s in load('sample_data') if s['sample_token'] in sample_scene]
    sd_index={s['token']:s for s in sd}
    for s in sd:
        for k in ('prev','next'):
            assert not s[k] or s[k] in sd_index,('sensor chain leaves selected scenes',s['token'],k)
    sensors=load('sensor'); channel={s['token']:s['channel'] for s in sensors}
    cs=load('calibrated_sensor'); needed_cs={s['calibrated_sensor_token'] for s in sd}
    cs=[s for s in cs if s['token'] in needed_cs]; cs_channel={s['token']:channel[s['sensor_token']] for s in cs}
    ego_tokens={s['ego_pose_token'] for s in sd}
    poses=[s for s in load('ego_pose') if s['token'] in ego_tokens]
    annotations=[s for s in load('sample_annotation') if s['sample_token'] in sample_scene]
    instance_tokens={s['instance_token'] for s in annotations}
    instances=[s for s in load('instance') if s['token'] in instance_tokens]
    common={n:load(n) for n in ('category','attribute','visibility','log','map')}
    audit={'created_at':time.time(),'raw_root':str(RAW),'scenes':[], 'protocol':{
        'id':'vehicle53_nuscenes_source_protected_blocks_v1',
        'rule':'Odd chronological sample-token blocks held out unless any retained production provenance uses that block; all other blocks train. All cameras and LiDAR in a block share the split.',
        'pose_metadata':'Native calibration, ego poses and actor trajectories available to scene methods; RGB/LiDAR observations in heldout blocks excluded from fitting and initialization.',
        'selection':'No model scores consulted; source-protection is conservative across all nested production provenance.',
        'model_selection':'Fixed final step30000; heldout metrics never choose checkpoints.',
        'vehicle_metrics':'Scene heldout frames require later target visibility/mask audit before asset scores; not automatically eligible GT surfaces.'}}
    for scene in scenes:
        name=scene['name']; ss=sorted([s for s in samples if s['scene_token']==scene['token']],key=lambda s:s['timestamp'])
        st={s['token'] for s in ss}; ds=[s for s in sd if s['sample_token'] in st]
        sensor_files=[s for s in ds if cs_channel[s['calibrated_sensor_token']].startswith('CAM_') or cs_channel[s['calibrated_sensor_token']]=='LIDAR_TOP']
        by_filename={s['filename']:s for s in sensor_files}
        unresolved=sorted(protected[name]-by_filename.keys())
        assert not unresolved,(name,'unresolved source paths',unresolved)
        protected_blocks={by_filename[p]['sample_token'] for p in protected[name]}
        heldout={s['token'] for i,s in enumerate(ss) if i%2==1 and s['token'] not in protected_blocks}
        assert heldout and st-heldout,(name,'empty split')
        files=[{'filename':s['filename'],'sample_data_token':s['token'],'sample_token':s['sample_token'],
                'timestamp_us':s['timestamp'],'channel':cs_channel[s['calibrated_sensor_token']],
                'is_key_frame':s['is_key_frame'],'split':'eval' if s['sample_token'] in heldout else 'train'} for s in sensor_files]
        missing=[f['filename'] for f in files if not (RAW/f['filename']).is_file() or (RAW/f['filename']).stat().st_size==0]
        counts=dict(collections.Counter(f['split']+'/'+f['channel'] for f in files))
        dest=R/'inputs/nuscenes'/name
        meta=dest/VERSION;meta.mkdir(parents=True,exist_ok=True)
        for sub in ('samples','sweeps','maps'):
            p=dest/sub
            if not p.exists():p.symlink_to(RAW/sub,target_is_directory=True)
        an=[s for s in annotations if s['sample_token'] in st];its={s['instance_token'] for s in an}
        cst={s['calibrated_sensor_token'] for s in ds};pt={s['ego_pose_token'] for s in ds}
        tables=dict(common,scene=[scene],sample=ss,sample_data=ds,sample_annotation=an,
                    instance=[s for s in instances if s['token'] in its],
                    calibrated_sensor=[s for s in cs if s['token'] in cst],
                    ego_pose=[s for s in poses if s['token'] in pt],sensor=sensors)
        for table,data in tables.items():write(meta/(table+'.json'),data)
        split=dict(protocol=audit['protocol'],scene=name,data_root=str(dest),protected_source_files=sorted(protected[name]),
                   protected_sample_tokens=sorted(protected_blocks),heldout_sample_tokens=sorted(heldout),files=files,counts=counts)
        split_path=R/'configs/nuscenes_splits'/(name+'.json');write(split_path,split)
        item={'scene':name,'status':'ready' if not missing else 'blocked_missing_sensor_files','missing_files':missing,
              'counts':counts,'protected_blocks':len(protected_blocks),'heldout_blocks':len(heldout),
              'split':str(split_path),'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
              'data_root':str(dest),'target_ids':[r['sample_id'] for r in rows if r['scene_id']==name],
              'metadata_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in meta.glob('*.json')}}
        audit['scenes'].append(item);write(R/'reports/nuscenes_training_preflight.json',audit)
        print(json.dumps({k:item[k] for k in ('scene','status','counts','protected_blocks','heldout_blocks')}),flush=True)
    audit['status']='completed';write(R/'reports/nuscenes_training_preflight.json',audit)
if __name__=='__main__': main()
