import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from test_muse_runtime import tiny_checkpoint
from muse.cli import main


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        tiny_checkpoint(self.root)

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["-m", str(self.root), "--ctx", "32", *args])
        return code, out.getvalue(), err.getvalue()

    def test_inspect_does_not_materialize_model_and_reports_budget(self):
        code, out, _ = self.run_cli("--inspect")
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertGreater(plan["expert_cache_bytes"], 0)
        self.assertGreater(plan["resident_bytes"], 0)

    def test_token_id_smoke_runs_prefill_and_generation(self):
        code, out, err = self.run_cli("--token-ids", "1,2,3", "--tokens", "3",
                                      "--prefill-chunk", "2", "--cache-gib", "0.0001")
        self.assertEqual(code, 0, err)
        self.assertEqual(len(json.loads(out)["generated_token_ids"]), 3)
        self.assertIn("read_bytes", err)

    def test_context_overflow_rejected_before_model_loading(self):
        code, _, err = self.run_cli("--token-ids", "1,2,3", "--tokens", "32")
        self.assertEqual(code, 1)
        self.assertIn("context", err.lower())

    def test_token_id_outside_vocab_is_rejected(self):
        code, _, err = self.run_cli("--token-ids", "999999", "--tokens", "1")
        self.assertEqual(code, 1)
        self.assertIn("vocabulary", err)

    def test_prompt_uses_checkpoint_template_without_forcing_missing_renderer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        core = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        tokenizer.save_pretrained(self.root)
        config = json.loads((self.root / "config.json").read_text())
        config["audio_config"] = {"out_hidden_size": 128}
        config["text_config"].update(bos_token_id=0, eos_token_id=1)
        (self.root / "config.json").write_text(json.dumps(config))
        code, _, err = self.run_cli("-p", "hello", "--tokens", "1")
        self.assertEqual(code, 0, err)


if __name__ == "__main__":
    unittest.main()
