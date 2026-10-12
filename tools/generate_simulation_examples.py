"""Regenerate demo JSON via the real OfflinePlan builder, never a JS path copy.

Only this explicit example generator supplies synthetic missing site values.
It does not save them to config/competition_template.json.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_sorting.offsite import load_site, build_offline_plan
from battery_sorting.simulation_export import export_offline_plan, write_plan


def generate():
    raw=json.loads((ROOT/'config/competition_template.json').read_text(encoding='utf-8'))
    geometry=raw['geometry']
    geometry['placement_contact_z_m']=geometry['pick_contact_z_m']
    geometry['workspace_bounds_m']=[-.5,.5,-.6,.2,-.05,.9]
    geometry['target_centers_base_xy_m']={
        'T0':[-.16,-.31],'P1':[-.07,-.38],'P2':[-.24,-.38],'P3':[-.155,-.20]}
    raw['calibration']['handeye_result']=str(ROOT/raw['calibration']['handeye_result'])
    site=load_site(raw)
    picks={'red':[.20,-.36],'yellow':[.11,-.36],'blue':[.20,-.26],'green':[.11,-.26]}
    tasks=[('basic',(c,)) for c in picks]+[('sequence',c) for c in itertools.permutations(picks,3)]
    index={}
    for task,colors in tasks:
        plan=build_offline_plan(site,task,colors,picks)
        bundle=export_offline_plan(plan,site,speeds=dict(transit=.05,approach=.02,contact=.005),
                                   acceleration=.1,data_kind='demo',label='Python 规划器 · 演示输入',
                                   object_centers={c:[*xy,.04] for c,xy in picks.items()})
        filename=task+'-'+ '-'.join(colors)+'.json'
        write_plan(bundle,ROOT/'web_sim/plans'/filename)
        index[task+':'+','.join(colors)]='./plans/'+filename
    write_plan(index,ROOT/'web_sim/plans/catalog.json')
    print(f'Generated {len(index)} plans using battery_sorting.offsite.build_offline_plan')


if __name__=='__main__':
    generate()
