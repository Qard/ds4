import unittest
from agent_trace_text import generated_code_text

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

if __name__=='__main__': unittest.main()
