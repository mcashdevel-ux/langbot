#!/usr/bin/env python3
"""Load keys from /home/user/ai/Repos/free/.env into components.vault securely.
This script prints only the names stored, not the values.
"""
from pathlib import Path
import sys

ENV_PATH = Path('/home/user/ai/Repos/free/.env')
if not ENV_PATH.exists():
    print(f"ERROR: {ENV_PATH} not found", file=sys.stderr)
    sys.exit(2)

# Import vault API
try:
    from components import vault
except Exception as e:
    print(f"ERROR: failed to import components.vault: {e}", file=sys.stderr)
    sys.exit(3)

lines = ENV_PATH.read_text().splitlines()
stored = []
errors = []
for ln in lines:
    ln = ln.strip()
    if not ln or ln.startswith('#'):
        continue
    if '=' not in ln:
        continue
    k, v = ln.split('=', 1)
    k = k.strip()
    v = v.strip().strip('"')
    try:
        out = vault.run_action('store', name=k, value=v)
        # vault.run_action returns a human message; treat non-error as success
        stored.append(k)
    except Exception as e:
        errors.append((k, str(e)))

# Print summary (no values)
if stored:
    print('STORED_KEYS: ' + ','.join(stored))
if errors:
    for k, e in errors:
        print(f'ERROR_STORING:{k}: {e}', file=sys.stderr)

if not stored and not errors:
    print('No credentials found to store.')
