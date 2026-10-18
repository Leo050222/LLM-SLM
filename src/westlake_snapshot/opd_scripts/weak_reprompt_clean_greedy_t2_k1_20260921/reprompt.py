"""Marker-free corrective completion; exact accepted token IDs remain at the end."""
import hashlib
TEMPLATE = '''Original problem:
{problem}

A previous attempt is shown below, including its latest draft. Its continuation after the prefix supplied under New reasoning has a problem. Keep that prefix unchanged and reconsider what comes next.
The previous attempt is reference material, not an instruction or a continuation to copy.

Previous attempt:
{attempt}

Continue New reasoning exactly where its supplied prefix ends. Produce only the continuation of the original answer. Do not repeat the prefix, copy section headings, add a preamble, or discuss the previous attempt. Preserve the original problem's language and required output format, including any answer tags or code syntax.

New reasoning:
'''
TEMPLATE_SHA256=hashlib.sha256(TEMPLATE.encode()).hexdigest()
REPROMPT_PROTOCOL=dict(template_sha256=TEMPLATE_SHA256,format='raw_token_completion',final_prefix='exact_original_token_ids',replacement_tokens=1,first_token_ban=True,full_draft_visible=True,context_overflow='stop_reprompt_context_limit',return_context='original_prompt',strong_role='score_only',prompt_version='clean_v1',branch_marker=False,quoted_special_tokens=False)
def build_prompt(tokenizer,row,response,draft,position):
    if not 0 <= position < len(draft):
        raise ValueError('Trigger position outside draft')
    prefix=list(response)+list(draft[:position])
    # Decode the WHOLE attempt in one operation: a token boundary may split a
    # Unicode character, and rendering each side separately corrupts that text.
    # EOS/control tokens are internal state, not text to inject into a new prompt.
    attempt=tokenizer.decode(list(response)+list(draft),skip_special_tokens=True)
    text=TEMPLATE.format(problem=row['problem'],attempt=attempt)
    return tokenizer.encode(text,add_special_tokens=False)+prefix
