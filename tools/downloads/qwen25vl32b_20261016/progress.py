#!/usr/bin/env python3
"""Read the persisted download and verification status without touching weights."""
import json
from pathlib import Path
from datetime import datetime, timezone

run = Path(__file__).resolve().parent
plan = json.loads((run/'plan.json').read_text())
result = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'model':plan['datasets'][0]['name']}
state = run/'status.json'
if state.exists():
    result.update(json.loads(state.read_text()))
transfer = Path(plan['root'])/'status.json'
if transfer.exists():
    d=json.loads(transfer.read_text())
    result.update(downloaded_bytes=d['downloaded_bytes'], expected_bytes=d['expected_bytes'],
                  percent=round(100*d['downloaded_bytes']/d['expected_bytes'],2),
                  files_complete=sum(x['state']=='COMPLETE' for x in d['files'].values()),
                  files_total=len(d['files']), download_updated_at=d['updated_at'],
                  download_state=d['state'])
final=Path(plan['publish_root'])/result['model']
result.update(published_path=str(final), verified=(final/'VERIFIED.json').exists())
print(json.dumps(result,indent=2))
