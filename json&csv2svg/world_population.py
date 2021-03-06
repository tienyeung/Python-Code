import json
import pygal.maps.world
from pygal.style import RotateStyle

from country_code import get_country_code


filename='/Users/apple/Desktop/download_file/population_data.json'
with open(filename) as f:
    pop_data=json.load(f)

cc_populations={}
for pop_dict in pop_data:
    if pop_dict['Year']=='2010':
        country_name=pop_dict['Country Name']
        population=int(float(pop_dict['Value']))
        code=get_country_code(country_name)
        if code:
            cc_populations[code]=population

#根据人口数量分组
cc_pop_1,cc_pop_2,cc_pop_3={},{},{}
for cc,pop in cc_populations.items():
    if pop < 10000000:
        cc_pop_1[cc]=pop
    elif pop < 1000000000:
        cc_pop_2[cc]=pop
    else:
        cc_pop_3[cc]=pop

wm_style=RotateStyle('#336699')

wm=wm=pygal.maps.world.World(style=wm_style)
wm.title='World population in 2010'
wm.add('0-10m',cc_pop_1)
wm.add('10m-1b',cc_pop_2)
wm.add('over 1b',cc_pop_3)

wm.render_to_file('newstyle_world_population_2010.svg')
