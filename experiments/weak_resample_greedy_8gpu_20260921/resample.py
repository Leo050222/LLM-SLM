"""Recompute current student logits at the exact original token prefix."""
import hashlib
TEMPLATE_SHA256=hashlib.sha256(b'original_prompt_plus_accepted_prefix_v1').hexdigest()
RESAMPLE_PROTOCOL=dict(template_sha256=TEMPLATE_SHA256,format='original_prompt_token_ids',final_prefix='exact_original_token_ids',replacement_tokens=1,first_token_ban=True,full_draft_visible=False,return_context='original_prompt',strong_role='score_only',logits='student_original_context_recomputed')
def build_prompt(tokenizer,row,response,draft,position):
    assert 0 <= position < len(draft)
    return list(row['prompt_token_ids'])+list(response)+list(draft[:position])
