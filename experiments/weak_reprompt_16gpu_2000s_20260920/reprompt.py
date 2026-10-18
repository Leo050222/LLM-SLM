"""Literal completion prompt; the final accepted prefix is appended as exact IDs."""
import hashlib
TEMPLATE = '''Original problem:
{problem}

The existing reasoning has a problem starting at the marked branching point. Keep the reasoning before that point and continue along a revised reasoning path.
The existing reasoning below includes the entire latest draft. The text after <BRANCH_POINT> is the old continuation to reconsider.

Existing reasoning:
{prefix}<BRANCH_POINT>{discarded}

Continue the unfinished New reasoning below exactly where it ends. Do not repeat its prefix or add a preamble. Follow the original problem's language and output requirements.

New reasoning:
'''
TEMPLATE_SHA256=hashlib.sha256(TEMPLATE.encode()).hexdigest()
REPROMPT_PROTOCOL=dict(template_sha256=TEMPLATE_SHA256,format='raw_token_completion',final_prefix='exact_original_token_ids',replacement_tokens=1,first_token_ban=True,full_draft_visible=True,context_overflow='stop_reprompt_context_limit',return_context='original_prompt',strong_role='score_only')
def build_prompt(tokenizer,row,response,draft,position):
    assert 0 <= position < len(draft)
    prefix=list(response)+list(draft[:position])
    text=TEMPLATE.format(problem=row['problem'],prefix=tokenizer.decode(prefix,skip_special_tokens=False),discarded=tokenizer.decode(draft[position:],skip_special_tokens=False))
    return tokenizer.encode(text,add_special_tokens=False)+prefix
