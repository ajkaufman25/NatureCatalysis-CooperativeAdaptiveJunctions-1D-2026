#!/usr/bin/env python3
"""Build SI Figure S15 from the diagram extracted literally from the SI.

Requires pdflatex with standalone, TikZ and Latin Modern. This schematic
has no numerical data. The checked PDF in figures/ can be used when TeX is
not installed. No silent replacement by an older schematic is performed.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile
import fitz

ROOT=Path(__file__).resolve().parents[1]

def build(outdir:Path):
    executable=shutil.which('pdflatex')
    if executable is None:raise RuntimeError('pdflatex is required to rebuild S15; use the included PDF otherwise.')
    source=ROOT/'docs/Figure_S15_model_framework.tex'
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='caj_s15_') as td:
        local=Path(td)/source.name;shutil.copy2(source,local)
        proc=subprocess.run([executable,'-interaction=nonstopmode','-halt-on-error',local.name],
                            cwd=td,capture_output=True,text=True,timeout=60)
        if proc.returncode:
            raise RuntimeError('S15 TeX build failed:\n'+proc.stdout[-4000:])
        pdf=outdir/'Figure_S15_model_framework.pdf'
        shutil.copy2(local.with_suffix('.pdf'),pdf)
        with fitz.open(pdf) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(str(pdf.with_suffix('.png')))
    return pdf

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'figures')
    args=parser.parse_args();print(build(args.outdir))

if __name__=='__main__':main()
