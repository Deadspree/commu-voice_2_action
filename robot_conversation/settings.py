from pathlib import Path

import yaml


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['server']['host'] not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Bind to loopback; use SSH forwarding for a remote GPU backend')
    checks = [(cfg['server']['port'], 1, 65535, 'server port'),
              (cfg['motion']['cpu_threads'], 1, 32, 'CPU threads'),
              (cfg['motion']['segment_seconds'], 1, 4, 'motion segment seconds'),
              (cfg['motion']['context_seconds'], 0, 1, 'motion context seconds'),
              (cfg['motion']['lookahead_seconds'], 0, 1, 'motion lookahead seconds'),
              (cfg['motion']['blend_seconds'], 0, .5, 'motion blend seconds'),
              (cfg['gemini']['max_output_tokens'], 1, 4096, 'response tokens'),
              (cfg['gemini']['connect_timeout_seconds'], 1, 60, 'connection timeout'),
              (cfg['gemini']['response_timeout_seconds'], 1, 300, 'response timeout'),
              (cfg['session']['history_turns'], 1, 20, 'history turns'),
              (cfg['session']['max_recording_seconds'], 1, 20, 'recording seconds'),
              (cfg['session']['max_reply_seconds'], 1, 30, 'response seconds')]
    for value, minimum, maximum, label in checks:
        if type(value) not in (int, float) or not minimum <= value <= maximum:
            raise ValueError(f'Invalid {label}: expected {minimum} to {maximum}')
    if not cfg['gemini']['model'] or not cfg['gemini']['voice']:
        raise ValueError('Gemini model and voice are required')
    return cfg
