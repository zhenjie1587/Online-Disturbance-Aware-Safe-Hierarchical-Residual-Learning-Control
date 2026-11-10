import hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
class OfflineChecks(unittest.TestCase):
    def test_configuration(self):
        from controller_config import CONFIG,participation
        self.assertEqual(CONFIG.comfort_schedule_offset,.5)
        self.assertEqual(participation(96,96,192,63),0)
        self.assertEqual(participation(288,96,192,63),1)
        self.assertEqual(participation(500,96,192,None),0)
    def test_rule_offset(self):
        from unified_rules import rule_teacher_target_temperature as f
        self.assertAlmostEqual(f(True,294.15,21,8,0,use_weather_correction=False),21.5)
        self.assertAlmostEqual(f(False,288.15,21,8,0,use_weather_correction=False),20.8)
    def test_source_freezes(self):
        for version in ['revision_v2','revision_v3']:
            folder=ROOT/'reproduction'/version
            frozen=json.loads((folder/'results/evaluation/source_freeze.json').read_text())
            for n,h in frozen.items():self.assertEqual(hashlib.sha256((folder/n).read_bytes()).hexdigest(),h)
    def test_single_run_dry_plan(self):
        s=subprocess.check_output([sys.executable,'-X','utf8',str(ROOT/'run.py'),'--dry-run'],text=True,encoding="utf-8",cwd=ROOT)
        self.assertEqual(json.loads(s)['decisions'],1344)
    def test_reproduction_dry_plans(self):
        for suite in ['main','extended','delay','immediate','apartment']:
            r=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/run_reproduction.py'),suite,'--dry-run'],capture_output=True,text=True,encoding="utf-8",cwd=ROOT)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_summary(self):
        import importlib.util,tempfile
        spec=importlib.util.spec_from_file_location('_summary',ROOT/'scripts/summarize_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)
            for i,x in enumerate([1.,3.]):
                p=folder/str(i);p.mkdir();(p/'result.json').write_text(json.dumps(dict(cost=x,energy=x,emissions=x,discomfort=x)))
            rows=m.summarize(folder);self.assertEqual(rows[0]['n'],2);self.assertEqual(rows[0]['mean'],2.)
if __name__=='__main__':unittest.main()
