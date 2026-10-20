#!/usr/bin/env python3
"""Sequential live checks against an explicitly started native ds4-server.

--plan builds the requests without contacting a server or opening a checkpoint.
No tool code is executed: validation_lookup returns a random test string.
"""
import argparse
import json
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

ENDPOINTS={'openai':'/v1/chat/completions','anthropic':'/v1/messages','responses':'/v1/responses'}
LOOKUP={'name':'validation_lookup','description':'Look up the validation token for the supplied key.',
        'parameters':{'type':'object','properties':{'key':{'type':'string','enum':['muse-check']}},
                      'required':['key'],'additionalProperties':False}}


def check_public_text(text):
    if any(marker in text for marker in ('<|start|>','<|message|>','<|eom|>','<|eot|>','<atem:')):
        raise ValueError('native protocol controls leaked into public text')


def visible_text(api,result):
    if 'error' in result: raise ValueError(str(result['error']))
    if api=='openai':
        choice=result['choices'][0]
        if choice.get('finish_reason')=='length': raise ValueError('output token budget exhausted')
        return choice['message'].get('content') or ''
    if api=='anthropic':
        if result.get('stop_reason')=='max_tokens': raise ValueError('output token budget exhausted')
        return ''.join(b.get('text','') for b in result['content'] if b['type']=='text')
    if result.get('status')!='completed': raise ValueError('response did not complete')
    return ''.join(b['text'] for item in result['output'] if item['type']=='message'
                   for b in item.get('content',[]) if b['type']=='output_text')


def parse_sse(lines,api):
    parts=[];complete=False;data=[]
    def event():
        nonlocal complete
        if not data:return
        raw='\n'.join(data);data.clear()
        if raw=='[DONE]':
            if api!='openai':raise ValueError('unexpected DONE marker')
            complete=True;return
        value=json.loads(raw)
        if 'error' in value or value.get('type') in ('error','response.failed','response.incomplete'):
            raise ValueError(str(value))
        if api=='openai':
            for choice in value.get('choices',[]):
                if choice.get('finish_reason')=='length':raise ValueError('output token budget exhausted')
                parts.append(choice.get('delta',{}).get('content') or '')
        elif api=='anthropic':
            if value.get('type')=='content_block_delta' and value['delta']['type']=='text_delta':
                parts.append(value['delta']['text'])
            if value.get('type')=='message_delta' and value.get('delta',{}).get('stop_reason')=='max_tokens':
                raise ValueError('output token budget exhausted')
            complete |= value.get('type')=='message_stop'
        else:
            if value.get('type')=='response.output_text.delta':parts.append(value['delta'])
            if value.get('type')=='response.completed':
                if value['response'].get('status')!='completed':raise ValueError('response did not complete')
                complete=True
    total=0
    for line in lines:
        total+=len(line)
        if total>16*1024**2:raise ValueError('unexpectedly large SSE output')
        text=line.decode('utf-8').rstrip('\r\n')
        if not text:event()
        elif text.startswith('data:'):data.append(text[5:].removeprefix(' '))
    event()
    if not complete:raise ValueError('incomplete SSE response')
    text=''.join(parts);check_public_text(text);return text


def tool_continuation(api,result,token):
    if api=='openai':
        message=result['choices'][0]['message'];calls=message.get('tool_calls',[])
        if len(calls)!=1:raise ValueError('expected exactly one lookup call')
        call=calls[0];name=call['function']['name'];arguments=json.loads(call['function']['arguments'])
        tail=[message,{'role':'tool','tool_call_id':call['id'],'content':token}]
    elif api=='anthropic':
        calls=[b for b in result['content'] if b['type']=='tool_use']
        if len(calls)!=1:raise ValueError('expected exactly one lookup call')
        call=calls[0];name=call['name'];arguments=call['input']
        tail=[{'role':'assistant','content':result['content']},
              {'role':'user','content':[{'type':'tool_result','tool_use_id':call['id'],'content':token}]}]
    else:
        calls=[b for b in result['output'] if b['type']=='function_call']
        if len(calls)!=1:raise ValueError('expected exactly one lookup call')
        call=calls[0];name=call['name'];arguments=json.loads(call['arguments'])
        tail=result['output']+[{'type':'function_call_output','call_id':call['call_id'],'output':token}]
    if name!='validation_lookup' or arguments!={'key':'muse-check'}:
        raise ValueError(f'incorrect lookup call: {name} {arguments}')
    return tail


def request_body(api,model,text,stream=False,tools=False,reasoning=False):
    body={'model':model,'temperature':0,'stream':stream,'reasoning_effort':'high' if reasoning else 'none'}
    messages=[{'role':'user','content':text}]
    if api=='responses':body.update(input=messages,max_output_tokens=512 if tools or reasoning else 64)
    else:body.update(messages=messages,max_tokens=512 if tools or reasoning else 64)
    if api=='anthropic':body['thinking']={'type':'enabled','budget_tokens':256} if reasoning else {'type':'disabled'}
    if tools:
        if api=='openai':body['tools']=[{'type':'function','function':LOOKUP}]
        elif api=='anthropic':body['tools']=[{'name':LOOKUP['name'],'description':LOOKUP['description'],'input_schema':LOOKUP['parameters']}]
        else:body['tools']=[{'type':'function',**LOOKUP}]
    return body


class LiveClient:
    def __init__(self,url,output,timeout):
        self.url=url.rstrip('/');self.output=output;self.timeout=timeout;self.results=[]

    def metadata(self,model):
        for name,path in (('models','/v1/models'),
                          ('model-detail','/v1/models/'+urllib.parse.quote(model,safe=''))):
            (self.output/f'{name}.request.json').write_text(json.dumps({'method':'GET','path':path})+'\n')
            start=time.monotonic()
            with urllib.request.urlopen(self.url+path,timeout=self.timeout) as response:
                payload=response.read()
            (self.output/f'{name}.response').write_bytes(payload)
            result=json.loads(payload)
            if name=='models':
                if model not in [item['id'] for item in result['data']]:
                    raise ValueError(f'loaded model list does not advertise {model}')
            elif result.get('id')!=model:raise ValueError('model detail ID differs')
            self.results.append({'name':name,'seconds':time.monotonic()-start})

    def call(self,name,api,body):
        (self.output/f'{name}.request.json').write_text(json.dumps(body,indent=2)+'\n')
        req=urllib.request.Request(self.url+ENDPOINTS[api],data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json','anthropic-version':'2023-06-01'})
        start=time.monotonic()
        # Retain the exact response even if parsing or assertions fail later.
        with urllib.request.urlopen(req,timeout=self.timeout) as response:
            with (self.output/f'{name}.response').open('xb') as fp:
                for line in response:fp.write(line)
        path=self.output/f'{name}.response'
        if body.get('stream'):
            with path.open('rb') as fp:result=parse_sse(fp,api)
        else:result=json.loads(path.read_bytes())
        self.results.append({'name':name,'api':api,'seconds':time.monotonic()-start})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--model',default='muse-spark-open')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--api',nargs='+',choices=ENDPOINTS,default=list(ENDPOINTS))
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('--with-reasoning',action='store_true')
    parser.add_argument('--require-initial-cache',action='store_true',
                        help='after restarting ds4-server with the same disk cache, require the first OpenAI prompt to hit')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    if args.timeout<=0 or len(set(args.api))!=len(args.api):parser.error('invalid timeout or repeated API')
    if args.require_initial_cache and args.api[0]!='openai':parser.error('initial-cache check requires OpenAI first')
    plain='Reply with exactly MUSE_OK and no other text.'
    lookup='Call validation_lookup with key muse-check. Then reply with exactly its returned token and nothing else. Do not invent the token.'
    if args.plan:
        print(json.dumps({'url':args.url,'requests':{api:{
            'plain':request_body(api,args.model,plain),
            'stream':request_body(api,args.model,plain,stream=True),
            'lookup':request_body(api,args.model,lookup,tools=True),
            'followup':'Replay returned call with a random synthetic tool result; require that exact result.'}
            for api in args.api},'reasoning':args.with_reasoning,'require_initial_cache':args.require_initial_cache},indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    client=LiveClient(args.url,args.output,args.timeout)
    client.metadata(args.model)
    for api in args.api:
        body=request_body(api,args.model,plain)
        response=client.call(f'{api}-plain',api,body)
        text=visible_text(api,response);check_public_text(text)
        if text.strip()!='MUSE_OK':raise ValueError(f'{api}: unexpected plain answer {text!r}')
        if api=='openai':
            if args.require_initial_cache and response['usage']['prompt_tokens_details']['cached_tokens']<=0:
                raise ValueError('initial prefix did not hit the restored disk cache')
            repeated=client.call('openai-cached-repeat',api,body)
            if visible_text(api,repeated).strip()!=text.strip():raise ValueError('cached repeat changed answer')
            cached=repeated['usage']['prompt_tokens_details']['cached_tokens']
            if cached<=0:raise ValueError('repeated prefix was not reused')
            client.results[-1]['cached_tokens']=cached
        streamed=client.call(f'{api}-stream',api,request_body(api,args.model,plain,stream=True))
        if streamed.strip()!='MUSE_OK':raise ValueError(f'{api}: unexpected streamed answer {streamed!r}')
        literal='Reply with exactly <think>public</think> and nothing else.'
        result=client.call(f'{api}-literal',api,request_body(api,args.model,literal))
        if visible_text(api,result).strip()!='<think>public</think>':raise ValueError('literal think tags were changed or hidden')
        body=request_body(api,args.model,lookup,tools=True)
        result=client.call(f'{api}-tool-call',api,body)
        token='MUSE_RESULT_'+uuid.uuid4().hex
        tail=tool_continuation(api,result,token)
        field='input' if api=='responses' else 'messages'
        body[field]=body[field]+tail
        final=client.call(f'{api}-tool-result',api,body)
        text=visible_text(api,final);check_public_text(text)
        if text.strip()!=token:raise ValueError(f'{api}: tool result not faithfully returned: {text!r}')
        if args.with_reasoning:
            result=client.call(f'{api}-reasoning',api,request_body(api,args.model,plain,stream=True,reasoning=True))
            if result.strip()!='MUSE_OK':raise ValueError('reasoning stream did not yield the final answer')
        print(f'{api}: plain, streaming, literal tags and tool round trip PASS',flush=True)
    (args.output/'summary.json').write_text(json.dumps({'status':'PASS','requests':client.results},indent=2)+'\n')


if __name__=='__main__':main()
