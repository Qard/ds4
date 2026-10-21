"""Check live agent traces and extract generated C without changing body tags."""
import re


def generated_code_text(raw,think,family):
    if family!='muse':
        if think:
            if '</think>' not in raw:raise ValueError('thinking did not finish')
            raw=raw.split('</think>',1)[1]
        return raw
    public=False;reading_header=True
    header='assistant';parts=[]
    for piece in re.split(r'(<\|(?:start|message|eom|eot|end_of_text)\|>)',raw):
        if piece=='<|start|>':
            public=False;reading_header=True;header=''
        elif piece=='<|message|>':
            if not reading_header:raise ValueError('message marker without a header')
            fields=header.split()
            recipients=[field[3:] for field in fields if field.startswith('to=')]
            if len(recipients)>1 or '' in recipients:raise ValueError('ambiguous recipient header')
            public=bool(fields) and fields[0]=='assistant' and (not recipients or recipients==['user'])
            reading_header=False;header=''
        elif piece in ('<|eom|>','<|eot|>','<|end_of_text|>'):
            public=False;reading_header=False;header=''
        elif reading_header:header+=piece
        elif public:parts.append(piece)
    if reading_header:raise ValueError('incomplete recipient header')
    return ''.join(parts)


def compaction_reused_system_prefix(trace,system_tokens):
    """Require successful rebuild work, with the complete system prefix cached.

    The tool-round sync has a different record shape and must not hide a cold
    compaction rebuild that happened immediately before it.
    """
    if system_tokens<=0:return False
    records=re.findall(r'^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ )?'
                       r'prefill sync done prompt=(\d+) cached=(\d+) suffix=(\d+) rc=(\d+)\b',trace,re.M)
    return bool(records) and all(rc==0 and cached>=system_tokens and suffix>0 and prompt==cached+suffix
               for prompt,cached,suffix,rc in (map(int,row) for row in records))
