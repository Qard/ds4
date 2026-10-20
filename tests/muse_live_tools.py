#!/usr/bin/env python3
"""Live two-call JSON/SSE and result-binding checks; no tool code is executed."""
import argparse
import copy
import json
import uuid
from pathlib import Path
from muse_live_api import LiveClient, request_body, visible_text, ENDPOINTS

PROMPT=('Call validation_lookup twice in this turn: once with key left and once with key right. '
        'Make both calls before waiting for results. After receiving both results, reply with exactly '
        'the left result, a vertical bar, then the right result, with no spaces or other text. Do not invent results.')


def multi_request_body(api,model,stream):
    body=copy.deepcopy(request_body(api,model,PROMPT,tools=True,stream=stream))
    body['max_output_tokens' if api=='responses' else 'max_tokens']=2048
    tool=body['tools'][0]
    schema=tool['function']['parameters'] if api=='openai' else tool['input_schema'] if api=='anthropic' else tool['parameters']
    schema['properties']['key']['enum']=['left','right']
    return body


def assemble_tool_stream(api,events):
    if api=='openai':
        calls={};content=[];finish=None
        for event in events:
            for choice in event.get('choices',[]):
                finish=choice.get('finish_reason') or finish;delta=choice.get('delta',{})
                if delta.get('content'):content.append(delta['content'])
                for part in delta.get('tool_calls',[]):
                    call=calls.setdefault(part['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                    if part.get('id'):call['id']+=part['id']
                    for key in ('name','arguments'):call['function'][key]+=part.get('function',{}).get(key,'')
        result={'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':''.join(content),'tool_calls':list(calls.values())}}]}
    elif api=='anthropic':
        blocks={};args={}
        for event in events:
            if event['type']=='content_block_start':blocks[event['index']]=event['content_block']
            if event['type']=='content_block_delta':
                delta=event['delta'];index=event['index']
                if delta['type']=='input_json_delta':args[index]=args.get(index,'')+delta['partial_json']
        for index,raw in args.items():blocks[index]['input']=json.loads(raw)
        result={'content':[v for v in blocks.values() if v['type']=='tool_use']}
    else:
        result=next((e['response'] for e in events if e.get('type')=='response.completed'),None)
        if result is None:raise ValueError('missing Responses completion')
    return result


def multi_continuation(api,result,values):
    if api=='openai':
        calls=result['choices'][0]['message'].get('tool_calls',[])
        normalized=[(t['id'],t['function']['name'],json.loads(t['function']['arguments'])) for t in calls]
    elif api=='anthropic':
        calls=[t for t in result['content'] if t['type']=='tool_use']
        normalized=[(t['id'],t['name'],t['input']) for t in calls]
    else:
        calls=[t for t in result['output'] if t['type']=='function_call']
        normalized=[(t['call_id'],t['name'],json.loads(t['arguments'])) for t in calls]
    if (len(normalized)!=2 or len({t[0] for t in normalized})!=2 or
        any(not t[0] or t[1]!='validation_lookup' or t[2] not in ({'key':'left'},{'key':'right'}) for t in normalized) or
        {t[2]['key'] for t in normalized}!={'left','right'}):
        raise ValueError(f'expected two distinct left/right lookup calls: {normalized!r}')
    if api=='openai':
        return [result['choices'][0]['message']]+[{'role':'tool','tool_call_id':i,'content':values[a['key']]} for i,n,a in normalized]
    if api=='anthropic':
        return [{'role':'assistant','content':result['content']},{'role':'user','content':[
            {'type':'tool_result','tool_use_id':i,'content':values[a['key']]} for i,n,a in normalized]}]
    return result['output']+[{'type':'function_call_output','call_id':i,'output':values[a['key']]} for i,n,a in normalized]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--model',default='muse-spark-open')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--timeout',type=float,default=3600)
    parser.add_argument('--api',nargs='+',choices=ENDPOINTS,default=list(ENDPOINTS))
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    if args.timeout<=0 or len(set(args.api))!=len(args.api):parser.error('invalid timeout or repeated API')
    if args.plan:
        print(json.dumps({api:{'json':multi_request_body(api,args.model,False),'stream':multi_request_body(api,args.model,True)} for api in args.api},indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    client=LiveClient(args.url,args.output,args.timeout)
    for api in args.api:
        for stream in (False,True):
            name=api+('-stream' if stream else '-json')
            body=multi_request_body(api,args.model,stream)
            result=client.call(name,api,body)
            if stream:
                events=[json.loads(line[5:]) for line in (args.output/(name+'.response')).read_text().splitlines()
                        if line.startswith('data:') and line[5:].strip()!='[DONE]']
                result=assemble_tool_stream(api,events)
            values={key:'NONCE_'+uuid.uuid4().hex for key in ('left','right')}
            tail=multi_continuation(api,result,values)
            follow=dict(body);follow['stream']=False
            key='input' if api=='responses' else 'messages'
            follow[key]=body[key]+tail
            text=visible_text(api,client.call(name+'-results',api,follow)).strip()
            if text!=values['left']+'|'+values['right']:raise ValueError(f'incorrect ordered tool results: {text!r}')
            (args.output/(name+'.parsed.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(name,'two calls and distinct bound results PASS',flush=True)
    (args.output/'summary.json').write_text(json.dumps({'status':'PASS','requests':client.results},indent=2)+'\n')


if __name__=='__main__':main()
