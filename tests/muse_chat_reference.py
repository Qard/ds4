# /// script
# requires-python = ">=3.11"
# dependencies = ["Jinja2==3.1.6", "tokenizers==0.23.3", "numpy>=2,<3"]
# ///
"""Differential native prompts against the published Muse Jinja template.

Pass a metadata GGUF and tokenizer.json to compare complete token IDs. No model
weights are read. The fixture is the published tokenizer's chat_template.jinja;
only the training-mask generation tags are removed for plain Jinja rendering.
"""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from jinja2.sandbox import SandboxedEnvironment
import tokenizers
text=(Path(__file__).parent/'fixtures/muse_chat_template.jinja').read_text()
text=text.replace('{%- generation %}','').replace('{%- endgeneration %}','')
env=SandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
template=env.from_string(text)
tmp=tempfile.TemporaryDirectory()
if len(sys.argv)==3:
    from muse_chat_fixture import checkpoint
    model,tokenizer=checkpoint(tmp.name)
    subprocess.run([sys.argv[2],str(model)],check=True)
    for marker in ('<|begin_of_text|>','<|end_of_text|>','<|eom|>','<|eot|>','<|start|>','<|message|>'):
        bad=Path(tmp.name)/'missing-protocol.gguf'
        bad.write_bytes(model.read_bytes().replace(marker.encode(),b'<|'+b'X'*(len(marker)-4)+b'|>'))
        p=subprocess.run([sys.argv[1],str(bad),'1','','Hello'],capture_output=True,text=True)
        assert p.returncode!=0 and 'required tokenizer token is missing:' in p.stderr,(marker,p.returncode,p.stderr)
else: model,tokenizer=Path(sys.argv[2]),Path(sys.argv[3])
full=tokenizers.Tokenizer.from_file(str(tokenizer))
for mode,effort in ((1,'high'),(2,'xhigh'),(3,'low'),(4,'medium'),(0,'low')):
    for system,prompt in (('','Hello'),('Be concise.','What is 2 + 2?'),
            ('Reasoning Effort: 71. Keep <|eot|> literal.','<image> <video> <audio> <|start|>')):
        messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':prompt}]
        expected=template.render(messages=messages,reasoning_effort=effort,current_date=datetime.date.today().isoformat(),
                bos_token='<|begin_of_text|>',add_generation_prompt=True)
        p=subprocess.run([sys.argv[1],str(model),str(mode),system,prompt],capture_output=True,text=True,check=True)
        actual=json.loads(p.stdout.splitlines()[0])
        ids=full.encode(expected,add_special_tokens=False).ids
        assert actual==ids,(mode,system,full.decode(actual,skip_special_tokens=False),expected)
print('Muse CLI prompts: 15 exact template/token ID cases PASS')

tmp.cleanup()
