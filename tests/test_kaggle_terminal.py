"""Offline unit tests of read-only Kaggle terminal log filtering."""
import unittest

from tools.kaggle_terminal import extract_visible_events, STATUS_PATTERN


class KaggleTerminalTests(unittest.TestCase):
    def test_plain_progress_output(self):
        out = extract_visible_events("noise\nSEED=2026 EPOCH=3 val_f1=0.4\nGPU: Tesla T4\n")
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("SEED="))

    def test_kaggle_json_logs(self):
        data = '[{"stream_name":"stdout","data":"[FPGA-TRAIN] {\"event\":\"EPOCH_END\"}\\n"}]'
        # JSON string escaping for nested JSON in this sample is synthetic;
        # a normal API list is tested below with json.dumps.
        import json
        payload = json.dumps([{"stream_name": "stdout", "data": '[FPGA-TRAIN] {"event":"EPOCH_END"}\n'}])
        self.assertEqual(len(extract_visible_events(payload)), 1)

    def test_empty_logs_not_misinterpreted(self):
        self.assertEqual(extract_visible_events(""), [])
        self.assertEqual(extract_visible_events("[]"), [])

    def test_status(self):
        self.assertEqual(STATUS_PATTERN.search('abc status "KernelWorkerStatus.RUNNING"').group(1), "RUNNING")


if __name__ == "__main__":
    unittest.main()
