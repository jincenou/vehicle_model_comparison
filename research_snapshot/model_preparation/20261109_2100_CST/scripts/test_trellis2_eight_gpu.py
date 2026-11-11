import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import dispatch_trellis2_vehicle432_eight_gpu as d

class EightGpuTests(unittest.TestCase):
    def test_live_adopted_generation_is_never_restarted(self):
        saved=dict(pid=12,start_ticks=123,state='R',command=['native'])
        with patch.object(d,'identity',return_value=saved):
            self.assertFalse(d.adoption_finished({},saved))

    def test_pid_reuse_rejected(self):
        saved=dict(pid=12,start_ticks=123,state='R',command=['native'])
        with patch.object(d,'identity',return_value=dict(saved,start_ticks=999)):
            with self.assertRaisesRegex(AssertionError,'PID identity'):d.adoption_finished({},saved)

    def test_exited_generation_requires_hashes_and_complete_state(self):
        job=dict(sample_id='a',generation_pid=12)
        generated=dict(pid=12,sample_id='a',status='running')
        with patch.object(d,'identity',return_value=None),patch.object(d.core,'read',return_value=generated):
            with self.assertRaisesRegex(AssertionError,'complete native'):d.adoption_finished(job,dict(pid=12))
        generated.update(status='generated_pending_reload',output_hashes={'native.pt':'good'},output='out',output_sha256='good')
        with patch.object(d,'identity',return_value=None),patch.object(d.core,'read',return_value=generated),patch.object(d.core,'sha',return_value='bad'):
            with self.assertRaises(AssertionError):d.adoption_finished(job,dict(pid=12))

    def test_reservation_does_not_override_actual_gpu_occupancy(self):
        detail=dict(uuid='GPU-test',free_mib=48000,reservations=[dict(source=str(d.RESERVATIONS[0]))])
        with patch.object(d.core,'available',return_value=(False,detail)),patch.object(d.subprocess,'check_output',return_value='GPU-test\n'):
            self.assertFalse(d.ready('0'))
        with patch.object(d.core,'available',return_value=(False,detail)),patch.object(d.subprocess,'check_output',return_value=''):
            self.assertTrue(d.ready('0'))
        detail['reservations'].append(dict(source='/other/job.json'))
        with patch.object(d.core,'available',return_value=(False,detail)),patch.object(d.subprocess,'check_output',return_value=''):
            self.assertFalse(d.ready('0'))

    def test_six_slots_adopt_and_reload_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'logs').mkdir();legacy=root/'assets';legacy.mkdir()
            rows=[dict(sample_id=f's{i}') for i in range(8)]
            jobs=[dict(**r,status='queued') for r in rows]
            jobs[0].update(status='running',gpu='0',generation_pid=99,child_pid=99)
            jobs[1].update(status='completed');jobs[2].update(status='failed',reason='retained')
            state=dict(jobs=jobs,signatures={},manifest_sha256='fixture')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(objects=rows)))
            old=root/'old.json';old.write_text(json.dumps(dict(jobs=[dict(**r,status='queued') for r in rows])))
            activity=[]
            def command(job,gpu):
                sid=job['sample_id']
                return [(s,[sys.executable,'-c','from pathlib import Path; Path('+repr(str(root/(sid+'_'+s)))+').write_text("once")'],{}) for s in d.prior.STAGES]
            def save(path,data):
                Path(path).write_text(json.dumps(data))
                if Path(path)==root/'state.json':activity.append(sum(j['status']=='running' for j in data['jobs']))
            with patch.object(d,'ROOT',root),patch.object(d,'STATE',root/'state.json'),patch.object(d,'RESERVATIONS',[root/f'reserve{i}.json' for i in range(6)]),\
                 patch.object(d.prior,'MANIFEST',manifest),patch.object(d.prior,'OLD_STATE',old),patch.object(d.prior,'EFFECTIVE',root/'effective.json'),patch.object(d.prior,'LEGACY',legacy),\
                 patch.object(d.core,'sha',return_value='fixture'),patch.object(d.core,'save',side_effect=save),patch.object(d.core.shutil,'disk_usage',return_value=type('Disk',(),dict(free=30*1024**3))()),\
                 patch.object(d,'acquire',side_effect=lambda gpu:(root/f'lease{gpu}').open('a')),patch.object(d,'adoption_finished',return_value=True),\
                 patch.object(d.prior,'commands',side_effect=command),patch.object(d.prior,'validate',return_value={'verified':True}),patch.object(d.time,'sleep',side_effect=lambda _:None):
                d.run(state,{'s0':dict(pid=99)},root)
            self.assertEqual(max(activity),6)
            self.assertFalse((root/'s0_generate').exists())
            self.assertTrue((root/'s0_native_reload').exists());self.assertTrue((root/'s0_glb_reload').exists())
            self.assertEqual(jobs[2]['status'],'failed');self.assertEqual(jobs[1]['status'],'completed')
            self.assertEqual(sum(j['status']=='completed' for j in jobs),7)

if __name__=='__main__':unittest.main()
