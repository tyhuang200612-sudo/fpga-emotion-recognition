"""Offline guardrails for complete monitored Kaggle training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.build_kaggle_monitor_v3 import build_monitor_v3
import tools.telemetry as tm


class MonitoredTrainingTests(unittest.TestCase):
    def test_notebook_has_actual_training_and_monitoring_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "kaggle-v3"
            md = build_monitor_v3(Path("."), dest)
            self.assertEqual(md["id"], "tyhuanga/fpga-tinycnn-monitor-v3")
            self.assertEqual(md["kernel_type"], "notebook")
            self.assertTrue(md["enable_gpu"])
            self.assertTrue(md["enable_internet"])
            self.assertEqual(md["dataset_sources"], ["msambare/fer2013"])
            notebook = json.loads((dest / md["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(notebook["cells"]), 7)
            self.assertTrue(notebook["metadata"]["kaggle"]["isInternetEnabled"])
            srcs = ["".join(c["source"]) for c in notebook["cells"][1:]]
            for i, src in enumerate(srcs):
                compile(src, f"v3-cell-{i}", "exec")
            self.assertIn("telemetry_selftest()", srcs[0])
            self.assertIn("torch.cuda.is_available()", srcs[1])
            self.assertIn("class TinyCNN", srcs[2])
            self.assertIn("def build_manifests(", srcs[3])
            self.assertIn('"EPOCH_END"', srcs[4])
            self.assertIn("run()", srcs[5])
            self.assertIn('publish_progress({"event":"ERROR"', srcs[5])
            self.assertEqual({f.name for f in dest.iterdir()},
                             {"kernel-metadata.json", md["code_file"]})

    @patch("tools.telemetry.urllib.request.urlopen")
    def test_outbound_data_allowlist_and_no_raw_secrets(self, urlopen):
        mock = MagicMock()
        mock.status = 200
        urlopen.return_value.__enter__.return_value = mock
        old_count,old_last=tm._TELEMETRY_SENT,tm._TELEMETRY_LAST
        try:
            tm._TELEMETRY_SENT=0
            tm._TELEMETRY_LAST=0
            self.assertTrue(tm.publish_progress(
                {"event":"EPOCH_END","epoch":1,"val_accuracy":0.66,
                 "sensitive_key":"DO_NOT_SEND","image_path":"/data/secret.png"}))
            req=urlopen.call_args.args[0]
            data=json.loads(req.data)
            self.assertEqual(data["epoch"],1)
            self.assertNotIn("image_path", data)
            self.assertNotIn("sensitive_key", data)
        finally:
            tm._TELEMETRY_SENT,tm._TELEMETRY_LAST=old_count,old_last


if __name__ == "__main__":
    unittest.main()
