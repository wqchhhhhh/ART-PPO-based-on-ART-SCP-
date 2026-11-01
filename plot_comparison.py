"""Plot dense physical replay and compare current RL with earlier baselines."""
import solver_environment
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cruise.config import Settings, Mission
from cruise.validation import replay
from cruise.dynamics import forces
from cruise.geometry import clearances

ROOT = Path(__file__).resolve().parent / '10302026' / 'results'

def main():
    names = [('PPO (new zones; infeasible)', 'rl_trajectory', '#d04c40'),
             ('IPOPT (previous; 3 zones)', 'local_expert', '#16759c'),
             ('SCP (archive; 3 zones)', 'cruise_scp', '#54994c')]
    runs=[]
    for label, stem, color in names:
        with np.load(ROOT / (stem+'.npz'), allow_pickle=False) as d:
            s=Settings.from_dict(json.loads(str(d['settings'])))
            m=Mission.from_dict(json.loads(str(d['mission'])))
            controls=d['controls'].copy()
        t,x,u,_=replay(s,m,controls)
        _,_,_,normal,q,heat,_=forces(x.T,u.T,s.vehicle)
        report=json.loads((ROOT/(stem+'.json')).read_text(encoding='utf-8-sig'))
        runs.append((label,color,s,m,t,x,u,q,heat,report,len(controls)))
    fig, axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    zones=runs[0][3].zones
    for ax in axes:
        for i,z in enumerate(zones):
            theta=np.linspace(0,2*np.pi,200); r=np.degrees(z.radius_km*1000/runs[0][2].vehicle.radius)
            ax.fill(z.longitude_deg+r*np.cos(theta)/np.cos(np.radians(z.latitude_deg)),z.latitude_deg+r*np.sin(theta),
                    color='#da6860' if i<3 else '#aa56b8',alpha=.25)
        ax.scatter(np.degrees(runs[0][3].initial[1]),np.degrees(runs[0][3].initial[2]),marker='o',c='black',label='Start')
        ax.scatter(np.degrees(runs[0][3].target[1]),np.degrees(runs[0][3].target[2]),marker='*',s=90,c='black',label='Target')
        ax.set(xlabel='Longitude [deg]',ylabel='Latitude [deg]');ax.grid(alpha=.2)
    for label,c,s,m,t,x,u,q,h,r,n in runs:
        axes[0].plot(np.degrees(x[:,1]),np.degrees(x[:,2]),color=c,label=label)
    x=runs[0][5]; axes[1].plot(np.degrees(x[:,1]),np.degrees(x[:,2]),color=runs[0][1],label='PPO dense replay')
    axes[0].set_title('Previous methods used only red zones');axes[1].set_title('PPO: 3 original + 2 purple random zones')
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle('Trajectory comparison | PPO failed feasibility; missions / grids differ')
    fig.savefig(ROOT/'trajectory_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    for label,c,s,m,t,x,u,q,h,r,n in runs:
        series=[x[:,0]/1000,x[:,3],q/1000,h/1e6,m.initial[6]-x[:,6],clearances(x,m,s.vehicle.radius).min(axis=1)/1000]
        for ax,values in zip(axs.flat,series):ax.plot(t,values,color=c,label=label)
    titles=['Altitude [km]','Speed [m/s]','Dynamic pressure [kPa]','Heat flux [MW/m2]','Fuel used [kg]','Min NFZ clearance [km]']
    for ax,title in zip(axs.flat,titles):ax.set(xlabel='Time [s]',ylabel=title);ax.grid(alpha=.2)
    for q in runs[0][2].limits.pressure:axs[1,0].axhline(q/1000,color='black',ls='--')
    axs[1,1].axhline(runs[0][2].limits.heat_max/1e6,color='black',ls='--')
    axs[2,1].axhline(0,color='black',ls='--');axs[0,0].legend(fontsize=8)
    fig.suptitle('Dense independent replay | negative clearance = NFZ penetration')
    fig.savefig(ROOT/'constraints_comparison.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
    for label,c,s,m,t,x,u,q,h,r,n in runs:ax.plot(np.degrees(x[:,1]),np.degrees(x[:,2]),x[:,0]/1000,color=c,label=label)
    ax.set(xlabel='Longitude [deg]',ylabel='Latitude [deg]',zlabel='Altitude [km]',title='3D trajectories (PPO infeasible)');ax.legend(fontsize=8)
    fig.savefig(ROOT/'trajectory_3d_comparison.png',dpi=170);plt.close(fig)
    rows=[]
    for label,c,s,m,t,x,u,q,h,r,n in runs:
        v=r['validation'];rows.append(dict(method=label,zones=len(m.zones),intervals=n,passed=v['passed'],fuel_kg=v['fuel_kg'],terminal_miss_m=v['terminal_miss_m'],pressure_min_kpa=v['pressure_range_pa'][0]/1000,pressure_max_kpa=v['pressure_range_pa'][1]/1000,heat_max_mw=v['heat_max_w_m2']/1e6,min_clearance_m=v['min_clearance_m'],runtime_s=r['runtime']))
    with (ROOT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
