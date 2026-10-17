"""C6 verbose retry: isolated Kaggle experiment and observable training trace."""
import json, tempfile, unittest
from pathlib import Path
from tools.build_kaggle_c6_verbose import build_c6_verbose

class C6VerboseTests(unittest.TestCase):
    def test_private_isolation_and_logging(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            result=build_c6_verbose(Path("."),folder)
            self.assertEqual(result["id"],"tyhuanga/fpga-c6-gap2-kd-verbose-retry1")
            self.assertTrue(result["is_private"] and result["enable_gpu"])
            self.assertEqual(set(result["dataset_sources"]),{
                "msambare/fer2013","tyhuanga/fpga-c2026-locked-baseline",
                "tyhuanga/fpga-c3-locked-teacher"})
            nb=json.loads((folder/result["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(nb["cells"]),8)
            trainer="".join(nb["cells"][6]["source"])
            last="".join(nb["cells"][7]["source"])
            for token in ("live-progress.jsonl", 'c6_progress("BATCH"',
                    'c6_progress("EPOCH_END"', 'c6_progress("TEACHER_LOADING_START"',
                    'C6_MODES=("GAP2_CE_control","GAP2_C3_KD")',
                    'C6_KD_WEIGHT=0.20', 'C6_SEEDS=(20261021,20261022)'):
                self.assertIn(token,trainer)
            self.assertIn("traceback.print_exc",last)
            self.assertIn("PYTHONUNBUFFERED",last)
            self.assertIn("run_c6()",last)
            self.assertNotIn('load_rows(manifest/"test.csv")',trainer)
            self.assertNotIn('load_rows(files/"test.csv")',trainer)
            for i,c in enumerate(nb["cells"]):
                if c["cell_type"]=="code":
                    compile("".join(c["source"]),f"c6-verbose-{i}","exec")

if __name__=="__main__":
    unittest.main()
