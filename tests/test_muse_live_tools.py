import json
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from muse_live_tools import assemble_tool_stream, multi_request_body, multi_continuation

class LiveToolsTests(unittest.TestCase):
    def test_openai_argument_fragments_and_result_ids(self):
        events=[{'choices':[{'delta':{'tool_calls':[
            {'index':0,'id':'call_l','function':{'name':'validation_lookup','arguments':'{"key":'}},
            {'index':1,'id':'call_r','function':{'name':'validation_lookup','arguments':'{"key":"right"}'}}]}}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"left"}'}}]},'finish_reason':'tool_calls'}]}]
        result=assemble_tool_stream('openai',events)
        tail=multi_continuation('openai',result,{'left':'VALUE_L','right':'VALUE_R'})
        self.assertEqual([(r['tool_call_id'],r['content']) for r in tail[1:]],[('call_l','VALUE_L'),('call_r','VALUE_R')])
        result['choices'][0]['message']['tool_calls'][1]['id']='call_l'
        with self.assertRaises(ValueError):multi_continuation('openai',result,{'left':'L','right':'R'})

    def test_anthropic_argument_fragments(self):
        events=[]
        for i,key in enumerate(('left','right')):
            events.extend([{'type':'content_block_start','index':i,'content_block':{'type':'tool_use','id':key,'name':'validation_lookup','input':{}}},
                {'type':'content_block_delta','index':i,'delta':{'type':'input_json_delta','partial_json':'{"key":'}},
                {'type':'content_block_delta','index':i,'delta':{'type':'input_json_delta','partial_json':json.dumps(key)+'}'}}])
        tail=multi_continuation('anthropic',assemble_tool_stream('anthropic',events),{'left':'L','right':'R'})
        self.assertEqual([r['content'] for r in tail[1]['content']],['L','R'])

    def test_responses_requires_completion_and_both_keys(self):
        with self.assertRaises(ValueError):assemble_tool_stream('responses',[])
        result={'output':[{'type':'function_call','call_id':k,'name':'validation_lookup','arguments':json.dumps({'key':k})} for k in ('right','left')]}
        tail=multi_continuation('responses',assemble_tool_stream('responses',[{'type':'response.completed','response':result}]),{'left':'L','right':'R'})
        self.assertEqual([r['output'] for r in tail[-2:]],['R','L'])
        result['output'][1]['arguments']='{"key":"right"}'
        with self.assertRaises(ValueError):multi_continuation('responses',result,{'left':'L','right':'R'})

    def test_reversed_output_only_results_preserve_ids_and_values(self):
        values={'left':'L','right':'R'}
        fixtures={
            'openai':{'choices':[{'message':{'role':'assistant','tool_calls':[
                {'id':k,'function':{'name':'validation_lookup','arguments':json.dumps({'key':k})}} for k in values]}}]},
            'anthropic':{'content':[{'type':'tool_use','id':k,'name':'validation_lookup','input':{'key':k}} for k in values]},
            'responses':{'output':[{'type':'function_call','call_id':k,'name':'validation_lookup','arguments':json.dumps({'key':k})} for k in values]}}
        for api,result in fixtures.items():
            with self.subTest(api=api):
                tail=multi_continuation(api,result,values,reverse=True,output_only=api!='openai')
                if api=='openai':
                    self.assertEqual([t['tool_call_id'] for t in tail[1:]],['right','left'])
                    self.assertEqual([t['content'] for t in tail[1:]],['R','L'])
                elif api=='anthropic':
                    self.assertEqual(len(tail),1)
                    self.assertEqual([t['tool_use_id'] for t in tail[0]['content']],['right','left'])
                    self.assertEqual([t['content'] for t in tail[0]['content']],['R','L'])
                else:
                    self.assertEqual(len(tail),2)
                    self.assertEqual([t['call_id'] for t in tail],['right','left'])
                    self.assertEqual([t['output'] for t in tail],['R','L'])
        with self.assertRaises(ValueError):multi_continuation('openai',fixtures['openai'],values,output_only=True)

    def test_plan_is_complete_and_does_not_contact_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'not-created'
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('muse_live_tools.py')),
                '--url','http://127.0.0.1:1','--output',str(output),'--plan'],check=True,capture_output=True,text=True)
            cases=json.loads(run.stdout)
            self.assertEqual(set(cases),{'openai','anthropic','responses'})
            for api,formats in cases.items():
                self.assertFalse(formats['json']['stream'])
                self.assertTrue(formats['stream']['stream'])
            self.assertFalse(output.exists())

    def test_multiple_schema_does_not_mutate_single_lookup(self):
        from muse_live_api import LOOKUP
        for api in ('openai','anthropic','responses'):
            multi_request_body(api,'muse-spark-open',True)
        self.assertEqual(LOOKUP['parameters']['properties']['key']['enum'],['muse-check'])

if __name__=='__main__':unittest.main()
