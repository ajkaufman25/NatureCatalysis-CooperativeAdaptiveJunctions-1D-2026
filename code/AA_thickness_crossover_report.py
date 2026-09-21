#!/usr/bin/env python3
"""Compatibility command for the current SI thickness figure S19.

The retired S20-S22 thickness layout is archived, not regenerated as current
SI numbering. This wrapper calls the common figure renderer, using a supplied
thickness table when requested. It never solves a BVP or modifies source data.
"""
from pathlib import Path
import argparse
import shutil
import tempfile
import pandas as pd
from publication_data import ROOT,convert_sweep
from plot_publication import make_figures

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/publication')
    parser.add_argument('--data-file',type=Path,help='Optional SI-named or original combined thickness CSV')
    parser.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output/thickness_figures')
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='caj_thickness_plot_') as td:
        work=Path(td);data=work/'data';data.mkdir()
        for file in (ROOT/'data/publication').glob('*.csv'):shutil.copy2(file,data/file.name)
        source=args.data_file
        if source is None:
            for name in ['AA_thickness_plot_data.csv','AA_thickness_sweep_combined_dense.csv']:
                candidate=args.data_dir/name
                if candidate.exists():source=candidate;break
        if source is None:raise FileNotFoundError('No thickness plotting table found')
        frame=convert_sweep(pd.read_csv(source))
        if 'source' in frame and (frame.source=='full').any():frame=frame[frame.source=='full']
        frame.to_csv(data/'AA_thickness_plot_data.csv',index=False)
        make_figures(data,work/'figures')
        for file in (work/'figures').glob('Figure_S19_AA_thickness_sensitivity.*'):
            shutil.copy2(file,args.outdir/file.name)
    print(f'Wrote current SI Figure S19 to {args.outdir}')

if __name__=='__main__':main()
