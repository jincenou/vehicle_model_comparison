"""Read-only evidence audit of failed frozen-cohort TRELLIS.2 jobs.

Writes a derived report only. Never changes job state, generation parameters,
acceptance gates, native output files, or the fixed cohort.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inspect(root):
    prep = root / 'model_preparation/20261109_2100_CST'
    cohort_path = root / 'inputs/rigid500_nusc100_v1/manifest.json'
    ids = {row['sample_id'] for row in read(cohort_path)['objects']}
    records = []
    snapshots = []
    for relative in ['runs/trellis2_vehicle432_parallel_v1/effective_batch_state.json',
                     'runs/trellis2_rigid68_v1/batch_state.json']:
        batch_path = prep / relative
        # Read one immutable byte snapshot despite a live controller's atomic updates.
        payload = batch_path.read_bytes()
        batch = json.loads(payload)
        snapshots.append(dict(path=str(batch_path), sha256=hashlib.sha256(payload).hexdigest(),
                              counts=batch['counts'], updated_at=batch.get('updated_at')))
        for job in batch['jobs']:
            if job['status'] != 'failed':
                continue
            sid = job['sample_id']
            assert sid in ids
            folder = Path(job.get('run_dir', prep / 'runs/trellis2_vehicle432_v1' / sid))
            state_path = folder / 'state.json'
            state = read(state_path)
            assert state['sample_id'] == sid
            if 'log' in job:
                log = Path(job['log'])
            elif job.get('producer') == 'supplement':
                log = prep / 'runs/trellis2_vehicle432_parallel_v1/logs' / (sid + '.log')
            else:
                log = prep / 'logs' / ('trellis2_' + sid + '.log')
            trace = log.read_text(errors='replace')
            image_path = Path(state['input'])
            with Image.open(image_path) as image:
                assert image.mode == 'RGBA'
                alpha = np.asarray(image.getchannel('A'))
                input_stats = dict(size=list(image.size), alpha_positive_pixels=int((alpha > 0).sum()),
                                   alpha_over_127_pixels=int((alpha > 127).sum()))
            row = dict(sample_id=sid, category=job['category'], queue_stage=job.get('stage'),
                       queue_status='failed', native_status=state['status'],
                       native_reason=state.get('reason'), state_path=str(state_path),
                       state_sha256=sha(state_path), log=str(log), log_sha256=sha(log),
                       input_sha256=sha(image_path), input_matches_generation_record=sha(image_path) == state['input_sha256'],
                       input_stats=input_stats, classification='unclassified_requires_inspection',
                       retry_authorized_by_audit=False, formal_metrics_qualified=False)
            if ('coords[:, 0].max()' in trace and 'input.numel() == 0' in trace
                    and state['status'] == 'failed' and state['stage'] == 'generate'):
                row.update(classification='generation_empty_sparse_coordinates',
                           native_mesh_exists=(folder / 'native_voxel_mesh.pt').exists(),
                           action='Keep generation failure; do not silently change seed, input, resolution or sampler')
            elif state['status'] == 'generated_pending_reload':
                hashes = {name: sha(folder / name) == expected
                          for name, expected in state['output_hashes'].items()}
                comparisons = []
                for i in range(2):
                    for kind in ['alpha', 'shaded', 'base_color']:
                        a = np.asarray(Image.open(folder / f'native_{kind}_{i}.png'))
                        b = np.asarray(Image.open(folder / f'reload_{kind}_{i}.png'))
                        assert a.shape == b.shape
                        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
                        comparisons.append(dict(view=i, kind=kind, max_difference=int(diff.max()),
                            mean_difference=float(diff.mean()), max_byte=int(a.max()),
                            nonzero_channels=int(np.count_nonzero(a)),
                            pixels_over_127=int((a.max(axis=2) > 127).sum())))
                row.update(output_hash_checks=hashes, render_comparisons=comparisons)
                exact = all(v['max_difference'] == 0 for v in comparisons)
                failed_support = any(v['nonzero_channels'] <= 100 for v in comparisons if v['kind'] == 'alpha')
                if all(hashes.values()) and exact and failed_support and 'np.count_nonzero(a)>100' in trace:
                    row.update(classification='serialized_output_exact_reload_low_alpha_support',
                               serialization_evidence='All recorded native output hashes match; six reload images identical',
                               action='Retain native output and failed support gate separately; no regeneration or quality-based cohort removal',
                               downstream_note='If representation and references later qualify, report degenerate output scores; do not silently drop it from paired reporting')
            records.append(row)
    assert len({r['sample_id'] for r in records}) == len(records)
    return dict(schema='trellis2_failure_evidence_v1', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                cohort_sha256=sha(cohort_path), batch_snapshots=snapshots,
                counts=dict(Counter(r['classification'] for r in records)), records=records,
                job_states_modified=False, outputs_modified=False, generation_retried=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect(args.workspace)
    output = args.output or args.workspace / 'reports/trellis2_failure_evidence_20261111.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(report=str(output), counts=result['counts'])))
