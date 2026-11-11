"""Native DriveStudio preprocessing for one frozen extension scene."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parents[1]
B=R/'inputs/benchmark_nuscenes100_v1'
REPO=Path(('/opt/rigid-benchmark/drivestudio'.replace('/opt/rigid-benchmark/drivestudio', __import__("os").environ.get('RIGID_DRIVESTUDIO', '/opt/rigid-benchmark/drivestudio')).replace('/opt/rigid-benchmark/external', __import__("os").environ.get('RIGID_EXTERNAL_ROOT', '/opt/rigid-benchmark/external'))))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',required=True);a=ap.parse_args()
    freeze=json.loads((B/'freeze.json').read_text())
    assert sha(B/'selection.json')==freeze['selection_sha256']
    row=next(x for x in json.loads((B/'selection.json').read_text())['scenes'] if x['scene']==a.scene and x['origin']=='new28')
    split=json.loads(Path(row['split']).read_text());assert sha(row['split'])==row['split_sha256']
    source=Path(row['data_root']);base=R/'inputs/drivestudio_nuscenes100_expansion'/a.scene
    processed=base/'processed/trainval/000';report=base/'input_audit.json'
    for name,digest in split['metadata_sha256'].items():assert sha(source/'v1.0-trainval'/name)==digest
    for f in split['files']:
        p=source/f['filename'];assert p.is_file() and p.stat().st_size==f['bytes'] and p.stat().st_mtime_ns==f['mtime_ns']
    assert not processed.exists(), 'Refuse to overwrite native processed data'
    sys.path.insert(0,str(REPO))
    from datasets.nuscenes.nuscenes_preprocess import NuScenesProcessor
    processor=NuScenesProcessor(load_dir=str(source),save_dir=str(base/'processed'),split='v1.0-trainval',interpolate_N=0,
        process_keys=['images','lidar','calib','dynamic_masks','objects'],process_id_list=[0],workers=1)
    processor.convert_one(0)
    images=list((processed/'images').glob('*.jpg'));lidars=list((processed/'lidar').glob('*.bin'))
    assert len(images)==row['train_samples']*6 and len(lidars)==row['train_samples']
    result={'status':'native_preprocess_completed','scene':a.scene,'source_split_sha256':row['split_sha256'],
        'data_root':str(source),'processed_path':str(processed),'processed_images':len(images),'processed_lidars':len(lidars),
        'interpolate_N':0,'real_heldout_eligible':False,'input_policy':'All six native keyframe cameras and LIDAR_TOP; complete pose/actor metadata disclosed.'}
    report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
