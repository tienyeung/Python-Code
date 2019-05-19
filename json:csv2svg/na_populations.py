import pygal.maps.world

wm=pygal.maps.world.World()
wm.title='population in USA'
wm.add('usa',{'ca':34128,'us':12345,'mx':23245})

wm.render_to_file('na_populalation.svg')
