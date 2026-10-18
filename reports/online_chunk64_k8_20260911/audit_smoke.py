import json
from pathlib import Path
import sys
root=Path(sys.argv[1]); results=root/'results'
status=json.loads((results/'run_status.json').read_text())
assert status['completed']==8 and not status['failed']
rows=[json.loads(p.read_text()) for p in sorted((results/'completed').glob('*.json'))]
assert len(rows)==8 and {r['dataset_index'] for r in rows}==set(range(8))
report=[]
for row in rows:
    cursor=0; teacher_count=0; takeovers=0
    for event in row['events']:
        assert event['response_offset']==cursor
        if 'bridge_token_ids' in event:
            assert event['judgments'][-1]['judgment']['select_anchor']
            assert all(not j['judgment']['select_anchor'] for j in event['judgments'][:-1])
            anchor=event['anchor_response_position']
            assert anchor>=cursor
            bridge=event['bridge_token_ids']
            assert bridge[0]!=event['judgments'][-1]['candidate']['student_token_id']
            assert len(bridge)<=event['requested_teacher_tokens']<=8
            assert row['response_token_ids'][anchor:anchor+len(bridge)]==bridge
            assert row['token_sources'][cursor:anchor]==['student']*(anchor-cursor)
            assert row['token_sources'][anchor:anchor+len(bridge)]==['teacher']*len(bridge)
            teacher_count+=len(bridge); takeovers+=1; cursor=anchor+len(bridge)
        else:
            assert all(not j['judgment']['select_anchor'] for j in event['judgments'])
            end=cursor+event['draft_tokens']
            assert row['token_sources'][cursor:end]==['student']*event['draft_tokens']
            cursor=end
    assert cursor==len(row['response_token_ids'])==len(row['token_sources'])<=1024
    assert teacher_count==row['token_sources'].count('teacher')
    report.append(dict(dataset_index=row['dataset_index'],tokens=cursor,takeovers=takeovers,
                       teacher_tokens=teacher_count,finish_reason=row['finish_reason']))
assert max(r['takeovers'] for r in report)>=2
value=dict(status='REAL_JUDGE_SMOKE_AUDIT_OK',rows=report,run_status=status,
           truncated_smoke=True,accuracy_claim=False)
(root/'smoke_audit.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value,indent=2))
