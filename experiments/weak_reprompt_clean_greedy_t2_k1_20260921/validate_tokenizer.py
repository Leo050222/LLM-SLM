"""CPU-only check with the actual Qwen tokenizer; no generation or GPU allocation."""
import json
from transformers import AutoTokenizer
from reprompt import build_prompt,TEMPLATE,TEMPLATE_SHA256
MODEL='/storage/yukaichengLab/share/model/Qwen3-1.7B'
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
row=dict(problem='Compute 2 + 3. Return the answer only.')
text='The result is 计算🙂 5.'
ids=t.encode(text,add_special_tokens=False)
e=ids+[t.eos_token_id]
for j in range(len(e)):
 out=build_prompt(t,row,[],e,j)
 head=TEMPLATE.format(problem=row['problem'],attempt=text)
 expected=t.encode(head,add_special_tokens=False)+e[:j]
 assert out==expected
 assert t.eos_token_id not in t.encode(head,add_special_tokens=False)
 # Before trigger j the exact original IDs must survive, including split-byte tokens.
 if j:assert out[-j:]==e[:j]
 assert '<BRANCH_POINT>' not in head
for split in range(len(ids)):
 out=build_prompt(t,row,ids[:split],e[split:],0)
 assert out==t.encode(TEMPLATE.format(problem=row['problem'],attempt=text),add_special_tokens=False)+ids[:split]
print(json.dumps(dict(passed=True,tokenizer=MODEL,template_sha256=TEMPLATE_SHA256,trigger_boundaries=len(e),response_draft_boundaries=len(ids),checks=['exact continuation IDs','whole Unicode decoding','quoted EOS removed','marker removed','no trailing message delimiter']),ensure_ascii=False))
