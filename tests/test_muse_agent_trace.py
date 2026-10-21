import unittest
from agent_trace_text import generated_code_text, compaction_reused_system_prefix

class AgentTraceTests(unittest.TestCase):
    def test_recipient_headers_and_reasoning_are_removed(self):
        raw=(' to=self<|message|>private planning<|eom|>'
             '<|start|>assistant to=commentary<|message|>Working.<|eom|>'
             '<|start|>assistant to=user<|message|>int value_0(void) { return 0; }<|eot|>')
        self.assertEqual(generated_code_text(raw,True,'muse'),'int value_0(void) { return 0; }')

    def test_forced_user_prefix_preserves_literal_tags(self):
        self.assertEqual(generated_code_text(' to=user<|message|>char *s="<think>literal</think>";<|eot|>',False,'muse'),
                         'char *s="<think>literal</think>";')
        self.assertEqual(generated_code_text('<think>hidden</think>int x;',True,'legacy'),'int x;')

    def test_incomplete_or_ambiguous_header_is_rejected(self):
        for raw in (' to=self',' to=user to=self<|message|>private',' to=<|message|>private'):
            with self.assertRaises(ValueError): generated_code_text(raw,True,'muse')

    def test_compaction_prefix_reuse_requires_a_complete_ledger(self):
        good='prefill sync done prompt=2317 cached=1341 suffix=976 rc=0 123.000 ms'
        self.assertTrue(compaction_reused_system_prefix(good,1341))
        self.assertTrue(compaction_reused_system_prefix('2026-10-21 08:19:19.551 '+good,1341))
        self.assertFalse(compaction_reused_system_prefix('2026-10-21 08:19:19.551 user=\"'+good+'\"',1341))
        cold='prefill sync done prompt=2317 cached=0 suffix=2317 rc=0 123.000 ms'
        self.assertFalse(compaction_reused_system_prefix(cold+'\n'+good,1341))
        for bad in (good.replace('cached=1341','cached=0'),
                    good.replace('suffix=976','suffix=100'),
                    good.replace('rc=0','rc=1'),
                    'prefill sync done tool_round=0 transcript=2319 prompt=2319 cached=2317 suffix=2'):
            self.assertFalse(compaction_reused_system_prefix(bad,1341))


if __name__=='__main__': unittest.main()
