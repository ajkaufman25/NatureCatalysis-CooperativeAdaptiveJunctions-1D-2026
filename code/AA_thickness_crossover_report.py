#!/usr/bin/env python3
"""Render SI Figures S20-S22 from the supplied, preserved thickness data.

Use AA_thickness_sweep.py separately to regenerate numerical solutions.
Characteristic depletion widths use native baseline bendings, not inferred
illuminated depletion boundaries. Band bending is center referenced.
"""
from pathlib import Path
import argparse
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cooperative_adaptive_junction_simulator import ModelParams, facet_from_mean_delta
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans','axes.spines.top':False,'axes.spines.right':False})

def scales():
 p=ModelParams();f=facet_from_mean_delta(.925,.15);eps=8.8541878128e-12*p.eps_r_s;q=1.602176634e-19;nd=f.ND_cm3*1e6;vt=8.617333262145e-5*p.T
 return dict(WH=math.sqrt(2*eps*f.band_bending_H_V/q/nd)*1e6,WO=math.sqrt(2*eps*f.band_bending_O_V/q/nd)*1e6,absorption=1e6/p.alpha_abs,Lp=math.sqrt(p.mu_p*vt*p.tau_p)*1e6)

def save(fig,out,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)

def regions(ax):
 ax.axvspan(.08,.5,color='#e4eaf0',alpha=.6,zorder=0)
 ax.axvspan(2,6,color='#ede5d8',alpha=.6,zorder=0)

def setup(ax):
 ax.set_xscale('log');ax.grid(alpha=.16,which='both');ax.set_xlim(.01,10)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,default=ROOT/'data'/'thickness');ap.add_argument('--data-file',type=Path,help='Optional combined or reproduced thickness CSV');ap.add_argument('--outdir',type=Path,default=ROOT/'reproduced_output'/'thickness_figures');args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(args.data_file if args.data_file is not None else args.data_dir/'AA_thickness_sweep_combined_dense.csv').sort_values('thickness_um');c=d[d.source=='full'];r=d[d.source=='refined'];s=scales();peak=d.loc[d.J_OWS_mA_cm2.idxmax()]
 fig,axs=plt.subplots(2,1,figsize=(6.6,6.3),sharex=True,layout='constrained')
 for ax,col,ylabel in zip(axs,['J_OWS_mA_cm2','contact_separation_V'],[r'$J_{\mathrm{OWS}}$ (mA cm$^{-2}$)',r'Operating $U_O-U_H$ (V)']):
  regions(ax);setup(ax);ax.plot(d.thickness_um,d[col],color='#205e87',lw=1.5);ax.plot(c.thickness_um,c[col],'o',ms=3,color='#205e87',label='Global sweep');ax.plot(r.thickness_um,r[col],'s',ms=2.7,color='#cf6726',label='Refined sweep');ax.set_ylabel(ylabel);ax.axvline(1,ls='--',lw=.9,color='.35');ax.axvline(peak.thickness_um,ls=':',lw=1,color='.35')
 axs[0].set_ylim(0,1.53);axs[0].legend(frameon=False,loc='upper left',fontsize=9);axs[0].annotate('Sampled maximum\n3.981 µm',xy=(peak.thickness_um,peak.J_OWS_mA_cm2),xytext=(.65,1.16),arrowprops={'arrowstyle':'-','lw':.7},fontsize=9)
 axs[1].set_ylim(1.64,1.94);axs[1].set_xlabel(r'STO thickness, $L$ (µm)');axs[1].text(1.07,1.665,'1 µm baseline',rotation=90,fontsize=9)
 for ax,label in zip(axs,'ab'):ax.text(-.12,1.03,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
 save(fig,args.outdir,'Figure_S20_AA_thickness_overview')
 fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained');regions(ax);setup(ax)
 for col,lab,color in [('Jgen_mA_cm2',r'$J_{\mathrm{gen}}$','#60666b'),('J_OWS_mA_cm2',r'$J_{\mathrm{OWS}}$','#205e87'),('counterflow_mA_cm2','Interfacial counterflow','#cf6726'),('SRH_mA_cm2','Bulk SRH loss','#7c4190')]:ax.plot(d.thickness_um,d[col],label=lab,color=color,lw=1.7)
 for x,lab in [(s['WH']+s['WO'],r'$W_H+W_O$'),(s['absorption'],r'$\alpha^{-1}$'),(s['Lp'],r'$L_{D,p}$')]:
  ax.axvline(x,color='.35',ls='--',lw=.8);ax.text(x*.93,3.06,lab,ha='right',va='top',fontsize=9)
 ax.set_ylim(0,3.25);ax.set_ylabel(r'Current density (mA cm$^{-2}$)');ax.set_xlabel(r'STO thickness, $L$ (µm)');ax.legend(loc='upper left',fontsize=9,frameon=False)
 save(fig,args.outdir,'Figure_S21_AA_thickness_current_budget')
 fig,axs=plt.subplots(2,1,figsize=(6.6,6.1),sharex=True,layout='constrained')
 for ax in axs:ax.set_xscale('log');ax.set_xlim(.047,.53);ax.grid(alpha=.16,which='both');ax.axvspan(.08,.5,color='#e4eaf0',alpha=.6,zorder=0)
 axs[0].plot(r.thickness_um,r.J_OWS_mA_cm2,'s-',color='#205e87',ms=3,lw=1.4);axs[0].set_ylabel(r'$J_{\mathrm{OWS}}$ (mA cm$^{-2}$)')
 for col,lab,color in [('band_bending_H_V','HER side','#205e87'),('band_bending_O_V','OER side','#cf6726')]:axs[1].plot(r.thickness_um,r[col],'s-',label=lab,color=color,ms=3,lw=1.4)
 axs[1].set_ylabel('Center-referenced\nband bending (V)');axs[1].set_xlabel(r'STO thickness, $L$ (µm)');axs[1].legend(frameon=False,loc='upper left',fontsize=9)
 for ax in axs:
  for x in [s['WH'],s['WO'],s['WH']+s['WO']]:ax.axvline(x,color='.4',ls='--',lw=.8)
 axs[0].text(s['WH']*.97,.61,r'$W_H$',rotation=90,ha='right',va='top',fontsize=9);axs[0].text(s['WO']*1.04,.61,r'$W_O$',rotation=90,ha='left',va='top',fontsize=9);axs[0].text((s['WH']+s['WO'])*1.03,.61,r'$W_H+W_O$',rotation=90,ha='left',va='top',fontsize=9)
 for ax,label in zip(axs,'ab'):ax.text(-.12,1.03,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
 save(fig,args.outdir,'Figure_S22_AA_thickness_shoulder')
if __name__=='__main__':main()
