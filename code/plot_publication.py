#!/usr/bin/env python3
"""Render current SI Figures S16-S19 from explicit energy/voltage CSVs.

This script never runs a semiconductor calculation or modifies its inputs.
Its panel arrangements, axes and labels follow the BA-clarified SI. Panels
are drawn separately and assembled as vector PDFs; PNG previews are also
written. S15 is generated from its literal TikZ source by build_framework.py.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
ROOT=Path(__file__).resolve().parents[1]


def make_figures(data_dir:Path,output_dir:Path):
    """Read SI-named columns; do not mix reference U columns with E energies."""
    DATA=Path(data_dir);OUT=Path(output_dir)
    OUT.mkdir(parents=True,exist_ok=True)
    required=['operating_states.csv','AA_operating_profile.csv','BA_operating_profile.csv',
              'native_band_bending_sweep.csv','common_transfer_rate_sweep.csv',
              'intensity_sweep.csv','AA_thickness_plot_data.csv']
    missing=[x for x in required if not (DATA/x).is_file()]
    if missing:raise FileNotFoundError(f'Missing publication CSV files: {missing}')
    with tempfile.TemporaryDirectory(prefix='caj_panels_') as temporary, plt.rc_context({
            'font.size':13,'axes.labelsize':13,'axes.titlesize':13.5,
            'legend.fontsize':11.3,'pdf.fonttype':42,'ps.fonttype':42}):
        PANELS=Path(temporary)
        def chart(title, xlabel, ylabel, figsize=(5.1,3.9)):
            fig=plt.figure(figsize=figsize)
            ax=fig.add_axes([.155,.18,.80,.72])
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
            ax.grid(alpha=.15)
            return fig,ax

        def save(fig,name):
            path=PANELS/f'{name}.pdf'
            fig.savefig(path);plt.close(fig)
            return path

        def assemble(name,files,cols=2):
            docs=[fitz.open(f) for f in files]
            w=max(d[0].rect.width for d in docs);h=max(d[0].rect.height for d in docs)
            rows=(len(docs)+cols-1)//cols
            out=fitz.open();page=out.new_page(width=cols*w,height=rows*h)
            for i,d in enumerate(docs):
                x=(i%cols)*w;y=(i//cols)*h
                page.show_pdf_page(fitz.Rect(x,y,x+w,y+h),d,0)
            out.save(OUT/f'{name}.pdf',garbage=4,deflate=True)
            pix=page.get_pixmap(matrix=fitz.Matrix(1.7,1.7));pix.save(str(OUT/f'{name}.png'))
            for d in docs:d.close()
            out.close()

        summary=pd.read_csv(DATA/'operating_states.csv').set_index('architecture')
        # S16: energy profiles. All energies reference the RHE electron level.
        files=[]
        for i,mode in enumerate(['AA','BA']):
            f=pd.read_csv(DATA/f'{mode}_operating_profile.csv');x=f.x_um
            r=summary.loc[mode];EH=r.E_cat_HER_eV;EO=r.E_cat_OER_eV
            fig,ax=chart(f'({chr(97+i)}) {mode}: finite-current OWS',r'Position, $x$ ($\mu$m)',
                         r'Electron energy, $E$ (eV)',figsize=(5.3,4.5))
            ax.set_position([.145,.24,.81,.66])
            for col,label,ls in [('E_CB_eV',r'$E_{\rm CB}$','-'),('E_VB_eV',r'$E_{\rm VB}$','-'),
                                 ('E_Fn_eV',r'$E_{F,n}$','--'),('E_Fp_eV',r'$E_{F,p}$','--')]:
                ax.plot(x,f[col],label=label,ls=ls,lw=1.7)
            ax.plot([-.075,0,np.nan,1,1.075],[EH,EH,np.nan,EO,EO],lw=3,
                     label=r'$E_{\rm cat}$')
            ax.plot([-.08,1.08,np.nan,-.08,1.08],[0,0,np.nan,-1.229,-1.229],ls=':',lw=1.3,
                     label=r'$E_{\rm redox}$')
            ax.set(xlim=(-.09,1.09),ylim=(-2.70,1.40),xticks=[0,.25,.5,.75,1])
            for pos in [0,1]:ax.axvline(pos,ls=':',lw=.6)
            ax.text(.5,.965,r'$E_{\rm RHE}=0$',transform=ax.transAxes,ha='center',va='top',fontsize=11.5)
            values=f'{r.J_gen_mA_cm2:.3g} = {r.J_OWS_mA_cm2:.3g} + {r.J_contact_mA_cm2:.3g} + {r.J_SRH_mA_cm2:.3g}'
            ax.text(.5,-.68,r'$J_{\rm gen}=J_{\rm OWS}+J_{\rm contact}+J_{\rm SRH}$'+'\n'+values,
                ha='center',va='center',fontsize=11.5,bbox=dict(boxstyle='round,pad=.4',fc='white',lw=.55))
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,
                      handlelength=2,columnspacing=1.1)
            files.append(save(fig,'S16'+mode))
        assemble('Figure_S16_AA_BA_operating_profiles',files)

        # S17: same native bending voltages; neither coordinate is an energy.
        g=pd.read_csv(DATA/'native_band_bending_sweep.csv')
        files=[]
        fig,ax=chart('(a) Differential native band bending',r'$\Delta V_{\rm bb}^{0}$ (V)',
                     r'$J_{\rm OWS}$ (mA cm$^{-2}$)')
        for mean in sorted(g.mean_V_bb_0_V.unique()):
            ss=g[np.isclose(g.mean_V_bb_0_V,mean)].sort_values('Delta_V_bb_0_V')
            ax.plot(ss.Delta_V_bb_0_V,ss.J_OWS_mA_cm2,marker='o',ms=3,label=f'{mean:g} V')
        ax.axvline(.15,ls=':',lw=1)
        ax.set_ylim(-.03,1.10);ax.legend(title=r'$\overline{V}_{\rm bb}^{0}$',loc='lower right',frameon=False,
                                       ncol=2,fontsize=11)
        files.append(save(fig,'S17a'))
        fig,ax=chart('(b) Mean native band bending',r'$\overline{V}_{\rm bb}^{0}$ (V)',
                     r'$J_{\rm OWS}$ (mA cm$^{-2}$)')
        ss=g[np.isclose(g.Delta_V_bb_0_V,.15)].sort_values('mean_V_bb_0_V')
        ax.plot(ss.mean_V_bb_0_V,ss.J_OWS_mA_cm2,marker='o',ms=4)
        ax.axvline(.925,ls=':',lw=1)
        ax.set_ylim(0,1.05);ax.text(.5,.12,r'$\Delta V_{\rm bb}^{0}=0.150$ V',transform=ax.transAxes,ha='center')
        files.append(save(fig,'S17b'))
        assemble('Figure_S17_native_asymmetry_sensitivity',files)

        # S18: current and energy separations. Energy differences are in eV;
        # the numerical catalyst-energy separation equals Delta Vcat in V.
        k=pd.read_csv(DATA/'common_transfer_rate_sweep.csv').sort_values('k_m_s')
        i=pd.read_csv(DATA/'intensity_sweep.csv').sort_values('intensity_mW_cm2')
        files=[]
        for panel,frame,xcol,xlabel,log in [('a',k,'k_m_s',r'$k_n=k_p$ (m s$^{-1}$)',True),
                                         ('c',i,'intensity_mW_cm2',r'365-nm intensity (mW cm$^{-2}$)',False)]:
            fig,ax=chart(f'({panel}) '+('Common transfer coefficient' if log else 'Illumination intensity'),
                         xlabel,r'Current density (mA cm$^{-2}$)')
            ax.plot(frame[xcol],frame.J_OWS_mA_cm2,marker='o',ms=4,label=r'$J_{\rm OWS}$')
            if log:ax.plot(frame[xcol],frame.J_contact_mA_cm2,marker='s',ms=4,label=r'$J_{\rm contact}$')
            else:ax.plot(frame[xcol],frame.J_gen_mA_cm2,marker='s',ms=4,label=r'$J_{\rm gen}$')
            if log:ax.set_xscale('log')
            ax.legend(frameon=False);files.append(save(fig,'S18'+panel))
            panel2=chr(ord(panel)+1)
            fig,ax=chart(f'({panel2}) Energy separations',xlabel,'Energy separation (eV)')
            ax.plot(frame[xcol],frame.Delta_E_cat_eV,marker='o',ms=4,
                    label=r'$E_{\rm cat,HER}-E_{\rm cat,OER}$')
            ax.plot(frame[xcol],frame.Delta_E_F_center_eV,marker='s',ms=4,
                    label=r'$E_{F,n}-E_{F,p}$ (center)')
            if log:ax.set_xscale('log')
            ax.set_ylim(1.55,2.95);ax.legend(frameon=False,loc='center',fontsize=11)
            files.append(save(fig,'S18'+panel2))
        assemble('Figure_S18_transfer_intensity_sensitivity',files)

        # S19: original global grid (31 stored operating points, no interpolation fit).
        t=pd.read_csv(DATA/'AA_thickness_plot_data.csv').sort_values('thickness_um')
        files=[]
        fig,ax=chart('(a) Steady-state current balance',r'STO thickness, $L$ ($\mu$m)',
                     r'Current density (mA cm$^{-2}$)')
        for c,label in [('J_gen_mA_cm2',r'$J_{\rm gen}$'),('J_OWS_mA_cm2',r'$J_{\rm OWS}$'),
                        ('J_contact_mA_cm2',r'$J_{\rm contact}$'),('J_SRH_mA_cm2',r'$J_{\rm SRH}$')]:
            ax.plot(t.thickness_um,t[c],label=label,lw=1.5)
        ax.axvline(1,ls=':',lw=1);ax.set(xscale='log',xlim=(.01,10),ylim=(0,3.2))
        ax.legend(loc='upper left',frameon=False);files.append(save(fig,'S19a'))
        fig,ax=chart('(b) Catalyst-potential separation',r'STO thickness, $L$ ($\mu$m)',
                     r'$\Delta V_{\rm cat}$ (V)')
        ax.plot(t.thickness_um,t.Delta_V_cat_V,lw=1.7,marker='o',ms=2.5)
        ax.axvline(1,ls=':',lw=1)
        ax.axhline(1.229,ls='--',lw=1,label='Reversible OWS: 1.229 V')
        ax.set(xscale='log',xlim=(.01,10),ylim=(1.15,2.02));ax.legend(frameon=False,loc='center',fontsize=11.5)
        files.append(save(fig,'S19b'))
        assemble('Figure_S19_AA_thickness_sensitivity',files)
        print(f'Rendered Figures S16-S19 from {DATA}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/publication')
    parser.add_argument('--outdir',type=Path,default=ROOT/'figures')
    args=parser.parse_args()
    make_figures(args.data_dir,args.outdir)

if __name__=='__main__':main()
