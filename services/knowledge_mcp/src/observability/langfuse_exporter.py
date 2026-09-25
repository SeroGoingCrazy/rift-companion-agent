"""Optional export of completed local evaluations; no SDK/network use by default."""
from __future__ import annotations

import math
import os
from typing import Any


def export_evaluation(samples: list[dict], results: list[dict], run_id: str,
                      enabled: bool = False, client: Any = None) -> dict:
    if not enabled:
        return {'status': 'disabled', 'exported': 0}
    if client is None:
        required = ['LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_BASE_URL']
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            return {'status': 'not_configured', 'missing': missing, 'exported': 0}
        from langfuse import Langfuse
        client = Langfuse()
    if not client.auth_check():
        return {'status': 'authentication_failed', 'exported': 0}
    by_key = {(r['variant'], r['id']): r for r in results}
    traces = []
    for sample in samples:
        result = by_key[(sample['variant'], sample['id'])]
        if any(s in ('pending', 'error') for s in result['status'].values()):
            continue
        # Replay timestamps describe export, not original query latency.
        with client.start_as_current_observation(
            as_type='span', name='textbook-ragas-evaluation', input=sample['user_input'],
            output=sample['response'], metadata={
                'run_id': run_id, 'question_id': sample['id'], 'variant': sample['variant'],
                'split': sample['split'], 'replayed': True,
                'contexts': sample['chunks'], 'metric_status': result['status'],
            },
        ) as span:
            for name, value in {**result['metrics'], **sample['ir']}.items():
                if value is not None and math.isfinite(value):
                    span.score(name=name, value=float(value), data_type='NUMERIC',
                               comment='Local Ragas evaluation or explicitly named page-level IR proxy.')
            traces.append(span.trace_id)
    client.flush()
    return {'status': 'submitted', 'exported': len(traces), 'trace_ids': traces,
            'note': 'SDK submission completed; verify ingestion in Langfuse before claiming server persistence.'}
