# Code x8 scoring integrity audit

2026-09-18: audited a fixed snapshot of 358,468 existing score files across all200 configuration/replicate groups. All snapshot entries passed source SHA256, scorer signature, dataset/problem/index, prompt, replicate/configuration, completed source status, Boolean result and pass/status consistency checks. **Zero integrity errors.** This is an integrity audit, not a re-execution of all tests or proof that every negative outcome is attributable to model code.

Generation timeouts:1,103, preserved as incorrect under the original900-second policy. LiveCodeBench:64 scoring execution timeouts and7 solution child crashes (all listed crash exitcodes are-11); MBPP:15 entries with official base/plus timeout status. These have valid score records and are counted incorrect by existing policy; not silently omitted. No low-load re-execution was performed in this audit, so infrastructure/load contribution to execution exceptions is not ruled out.

Scoring was still running during the audit. The snapshot does not cover scores created after snapshot collection, and absence of final completion marker means all463,400 generated outputs had not yet been certified scored. Use audit.json for per-dataset counts and exception examples; audit.py contains exact checks.
