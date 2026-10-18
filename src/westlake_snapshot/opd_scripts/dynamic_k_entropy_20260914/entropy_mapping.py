"""Strict vLLM0.23 internal-to-external ID mapping for LLM's numeric IDs."""
import re


def external_records(records):
    result = {}
    for internal_id, value in records.items():
        match = re.fullmatch(r'([0-9]+)-[0-9a-f]{8}', internal_id)
        if match is None:
            raise ValueError('Unsupported vLLM entropy request ID format: ' + internal_id)
        external_id = match.group(1)
        if external_id in result:
            raise ValueError('Ambiguous entropy request ID mapping')
        result[external_id] = dict(value, internal_request_id=internal_id)
    return result
