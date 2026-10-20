import io
import json
import unittest
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from muse_live_api import parse_sse, visible_text, tool_continuation, check_public_text, check_no_reasoning, request_body, requested_effort

class LiveApiTests(unittest.TestCase):
    def test_sse_channels_and_completion(self):
        wire=(b': ping\r\n\r\ndata: {"choices":[{"delta":{"reasoning_content":"private"}}]}\r\n\r\n'
              b'data: {"choices":[{"delta":{"content":"MUSE_"}}]}\n\n'
              b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n')
        self.assertEqual(parse_sse(io.BytesIO(wire),'openai'),'MUSE_OK')
        with self.assertRaisesRegex(ValueError,'incomplete'):
            parse_sse(io.BytesIO(wire.replace(b'data: [DONE]\n\n',b'')),'openai')
        with self.assertRaises(ValueError):
            parse_sse(io.BytesIO(b'data: {"error":{"message":"bad"}}\n\n'),'openai')

    def test_disabled_reasoning_cannot_leak(self):
        result={'choices':[{'message':{'reasoning_content':'private'}}]}
        with self.assertRaises(ValueError):check_no_reasoning('openai',result)
        wire=b'data: {"choices":[{"delta":{"reasoning_content":"private"}}]}\n\ndata: [DONE]\n\n'
        with self.assertRaises(ValueError):parse_sse(io.BytesIO(wire),'openai',allow_reasoning=False)

    def test_reasoning_probe_uses_native_sampling(self):
        body=request_body('openai','muse-spark-open','Reply MUSE_OK',reasoning=True)
        self.assertEqual(body['temperature'],1)
        self.assertEqual(body['top_p'],1)
        self.assertEqual(body['min_p'],0)
        self.assertGreaterEqual(body['max_tokens'],1024)

    def test_responses_uses_nested_reasoning_effort(self):
        for enabled in (False,True):
            body=request_body('responses','muse-spark-open','Reply MUSE_OK',reasoning=enabled)
            self.assertNotIn('reasoning_effort',body)
            self.assertEqual(body['reasoning']['effort'],'high' if enabled else 'none')

    def test_other_api_streams(self):
        cases=[('anthropic',[
            {'type':'content_block_delta','delta':{'type':'thinking_delta','thinking':'hidden'}},
            {'type':'content_block_delta','delta':{'type':'text_delta','text':'MUSE_OK'}},
            {'type':'message_stop'}]),('responses',[
            {'type':'response.reasoning_summary_text.delta','delta':'hidden'},
            {'type':'response.output_text.delta','delta':'MUSE_OK'},
            {'type':'response.completed','response':{'status':'completed'}}])]
        for api,events in cases:
            wire=''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode()
            self.assertEqual(parse_sse(io.BytesIO(wire),api),'MUSE_OK')

    def test_tool_result_is_tied_to_call(self):
        response={'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
            'tool_calls':[{'id':'call_a','type':'function','function':{'name':'validation_lookup','arguments':'{"key":"muse-check"}'}}]}}]}
        tail=tool_continuation('openai',response,'nonce_817')
        self.assertEqual(tail[-1],{'role':'tool','tool_call_id':'call_a','content':'nonce_817'})
        response['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"key":"wrong"}'
        with self.assertRaises(ValueError): tool_continuation('openai',response,'nonce_817')

    def test_channel_controls_never_become_public_text(self):
        check_public_text('Literal <think>public</think>')
        for text in ('<|start|>assistant to=self','<atem:function_calls>'):
            with self.assertRaises(ValueError): check_public_text(text)
        result={'status':'completed','output':[{'type':'reasoning','summary':[{'text':'private'}]},
                          {'type':'message','content':[{'type':'output_text','text':'MUSE_OK'}]}]}
        self.assertEqual(visible_text('responses',result),'MUSE_OK')

class LiveHttpHarnessTests(unittest.TestCase):
    def test_complete_client_workflow_without_a_checkpoint(self):
        received=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                result={'id':'muse-spark-open'}
                if self.path=='/v1/models':result={'data':[result]}
                payload=json.dumps(result).encode();self.send_response(200)
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(body)
                api='responses' if self.path.endswith('/responses') else 'anthropic' if self.path.endswith('/messages') else 'openai'
                messages=body.get('messages',body.get('input'))
                last=messages[-1]
                is_result=(last.get('role')=='tool' or last.get('type')=='function_call_output' or isinstance(last.get('content'),list))
                if is_result:
                    if api=='openai':
                        self.server.test.assertEqual(last['tool_call_id'],'call_a');text=last['content']
                    elif api=='anthropic':
                        self.server.test.assertEqual(last['content'][0]['tool_use_id'],'call_a');text=last['content'][0]['content']
                    else:
                        self.server.test.assertEqual(last['call_id'],'call_a');text=last['output']
                else: text='<think>public</think>' if '<think>' in last['content'] else 'MUSE_OK'
                tool=bool(body.get('tools')) and not is_result
                call={'id':'call_a','type':'function','function':{'name':'validation_lookup','arguments':'{"key":"muse-check"}'}}
                if api=='openai':
                    message={'role':'assistant','content':None if tool else text,'reasoning_content':'private'}
                    if tool:message['tool_calls']=[call]
                    result={'choices':[{'message':message,'finish_reason':'tool_calls' if tool else 'stop'}],
                            'usage':{'prompt_tokens_details':{'cached_tokens':12}}}
                    events=[{'choices':[{'delta':{'reasoning_content':'private'}}]},
                            {'choices':[{'delta':{'content':text[:3]}}]},
                            {'choices':[{'delta':{'content':text[3:]},'finish_reason':'stop'}]}]
                elif api=='anthropic':
                    content=[{'type':'tool_use','id':'call_a','name':'validation_lookup','input':{'key':'muse-check'}}] if tool else [{'type':'text','text':text}]
                    result={'content':content,'stop_reason':'tool_use' if tool else 'end_turn'}
                    events=[{'type':'content_block_delta','delta':{'type':'thinking_delta','thinking':'private'}},
                            {'type':'content_block_delta','delta':{'type':'text_delta','text':text}}, {'type':'message_stop'}]
                else:
                    output=[{'type':'function_call','id':'fc_a','call_id':'call_a','name':'validation_lookup','arguments':'{"key":"muse-check"}'}] if tool else [{'type':'message','content':[{'type':'output_text','text':text}]}]
                    result={'status':'completed','output':output}
                    events=[{'type':'response.reasoning_summary_text.delta','delta':'private'},
                            {'type':'response.output_text.delta','delta':text},
                            {'type':'response.completed','response':result}]
                if requested_effort(body)=='none':
                    events=events[1:]
                    if api=='openai':result['choices'][0]['message'].pop('reasoning_content',None)
                if body.get('stream'):
                    data=''.join('data: '+json.dumps(e)+'\n\n' for e in events)
                    if api=='openai':data+='data: [DONE]\n\n'
                    payload=data.encode();kind='text/event-stream'
                else:payload=json.dumps(result).encode();kind='application/json'
                self.send_response(200);self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(payload)));self.end_headers()
                # Deliberately fragment JSON and SSE lines at transport level.
                for i in range(0,len(payload),7):self.wfile.write(payload[i:i+7]);self.wfile.flush()
        server=HTTPServer(('127.0.0.1',0),Handler);server.test=self
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'results'
                subprocess.run([sys.executable,str(Path(__file__).with_name('muse_live_api.py')),
                    '--url',f'http://127.0.0.1:{server.server_port}','--output',str(out),
                    '--with-reasoning','--require-initial-cache'],check=True,capture_output=True,timeout=20)
                summary=json.loads((out/'summary.json').read_text())
                self.assertEqual(summary['status'],'PASS')
                self.assertTrue((out/'models.response').exists())
                self.assertTrue((out/'model-detail.response').exists())
                self.assertEqual(len(summary['requests']),21)
                self.assertEqual(len(received),19)
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
