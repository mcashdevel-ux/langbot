#!/usr/bin/env python3
"""Compare variable names in ../free/.env with names stored in components.vault.
Prints only names and a simple report; does not reveal secret values.
"""
from pathlib import Path
import sys

ENV_PATH = Path('/home/user/ai/Repos/free/.env')
if not ENV_PATH.exists():
    print(f"ERROR: {ENV_PATH} not found", file=sys.stderr)
    sys.exit(2)

lines = ENV_PATH.read_text().splitlines()
env_names = []
for ln in lines:
    ln = ln.strip()
    if not ln or ln.startswith('#'):
        continue
    if '=' not in ln:
        continue
    k, _ = ln.split('=', 1)
    env_names.append(k.strip())

# Import vault API
try:
    from components import vault
except Exception as e:
    print(f"ERROR: failed to import components.vault: {e}", file=sys.stderr)
    sys.exit(3)

# Get vault list output
try:
    out = vault.run_action('list')
except Exception as e:
    print(f"ERROR: vault.list failed: {e}", file=sys.stderr)
    sys.exit(4)

# vault.run_action('list') returns a string; parse for names (assume one per line)
vault_names = []
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    # Heuristic: names are typically uppercase tokens without spaces; skip headings
    if line.lower().startswith('no credentials'):
        continue
    # extract words that look like NAMES
    parts = [p.strip() for p in line.split() if p.strip()]
    for p in parts:
        if p.isidentifier() and p.upper()==p:
            vault_names.append(p)

# Deduplicate
env_names = list(dict.fromkeys(env_names))
vault_names = list(dict.fromkeys(vault_names))

print('ENV_NAMES_COUNT:', len(env_names))
print('ENV_NAMES:', ','.join(env_names))
print('\nVAULT_NAMES_COUNT:', len(vault_names))
print('VAULT_NAMES:', ','.join(vault_names))

missing = [n for n in env_names if n not in vault_names]
extra = [n for n in vault_names if n not in env_names]
print('\nMISSING_IN_VAULT_COUNT:', len(missing))
print('MISSING_IN_VAULT:', ','.join(missing))
print('\nEXTRA_IN_VAULT_COUNT:', len(extra))
print('EXTRA_IN_VAULT:', ','.join(extra))
