#!/usr/bin/env python3
"""Resume pinned public model files, verify them offline, and publish on H800."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

RUN = Path(__file__).resolve().parent


def status(state, **values):
    data = {'state': state, 'pid': os.getpid(), 'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}
    temporary = RUN / 'status.json.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, RUN / 'status.json')


def main():
    with (RUN / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((RUN / 'plan.json').read_text())
        model = plan['datasets'][0]
        final = Path(plan['publish_root']) / model['name']
        if final.exists():
            source = final / 'SOURCE.json'
            if not source.is_file() or json.loads(source.read_text()).get('revision') != model['revision']:
                raise RuntimeError('Refusing to replace an existing model directory')
        else:
            spec = importlib.util.spec_from_file_location('resumable_download', RUN / 'download.py')
            downloader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(downloader)
            original_routes = downloader.routes
            route_file = RUN / 'routes.json'
            route_config = json.loads(route_file.read_text()) if route_file.exists() else {}
            alternatives = {entry['url']: entry.get('alternate_urls', []) for entry in model['files']}

            def routes(url):
                parsed = urlsplit(url)
                mirror = urlunsplit((parsed.scheme, 'hf-mirror.com', parsed.path, parsed.query, parsed.fragment))
                known = {'hf_proxy': (url, True), 'mirror_direct': (mirror, False),
                         'mirror_proxy': (mirror, True), 'hf_direct': (url, False)}
                for alternate in alternatives.get(url, []):
                    known['modelscope_direct'] = (alternate, False)
                    known['modelscope_proxy'] = (alternate, True)
                order = route_config.get('order', ['hf_proxy', 'mirror_direct', 'mirror_proxy', 'hf_direct'])
                selected = [known[key] for key in order if key in known]
                selected += [value for value in original_routes(url) if value not in selected]
                return selected

            downloader.routes = routes
            downloader.WORKERS = 8
            status('DOWNLOADING', model=model['name'], revision=model['revision'],
                   expected_bytes=sum(item['size'] for item in model['files']),
                   progress_path=str(Path(plan['root']) / 'status.json'))
            sys.argv = [str(RUN / 'download.py'), '--plan', str(RUN / 'plan.json')]
            result = downloader.main()
            if result:
                status('DOWNLOAD_ERROR', exit_code=result)
                return result
        status('VERIFYING', model=model['name'])
        deadline = time.monotonic() + 300
        while not (RUN / 'verify_models.py').is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError('Verifier has not been deployed')
            time.sleep(5)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        result = subprocess.run([sys.executable, '-u', str(RUN / 'verify_models.py'),
                                 '--plan', str(RUN / 'plan.json')], env=env)
        status('COMPLETE' if result.returncode == 0 else 'VERIFY_ERROR',
               model=model['name'], verification_exit_code=result.returncode,
               publish_path=str(final))
        return result.returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        status('ERROR', error_type=type(error).__name__, error=str(error))
        raise
