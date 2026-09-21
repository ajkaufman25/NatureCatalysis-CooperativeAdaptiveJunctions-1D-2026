#!/usr/bin/env python3
"""Verify delivered file sizes/SHA-256 checksums, or deliberately rebuild them.

Normal invocation is read-only. --write replaces MANIFEST_SHA256.csv after a
reviewed local revision. The manifest excludes itself and transient outputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','__pycache__','reproduced_output','redrawn_figures'}

def files():
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in EXCLUDED for x in rel.parts):continue
        if p.name=='MANIFEST_SHA256.csv' or p.suffix in ('.pyc','.pyo'):continue
        yield p,rel.as_posix()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args();manifest=ROOT/'MANIFEST_SHA256.csv'
    if args.write:
        with manifest.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['relative_path','size_bytes','sha256'])
            for p,rel in files():w.writerow([rel,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
        print('Wrote '+str(manifest));return
    if not manifest.exists():raise SystemExit('Missing MANIFEST_SHA256.csv')
    rows=list(csv.DictReader(manifest.open()));failed=[];listed=set()
    for row in rows:
        rel=row['relative_path'];p=ROOT/rel;listed.add(rel)
        if not p.is_file():failed.append(rel+' missing');continue
        if p.stat().st_size!=int(row['size_bytes']) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            failed.append(rel+' checksum/size mismatch')
    extra=set(rel for _,rel in files())-listed
    failed += [x+' unlisted' for x in sorted(extra)]
    if failed:raise SystemExit('\n'.join(failed))
    print(f'Integrity check passed for {len(rows)} files')

if __name__=='__main__':main()
