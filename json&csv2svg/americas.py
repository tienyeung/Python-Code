import pygal.maps.world

wm=pygal.maps.world.World()
wm.title='North,Central,and South USA'

wm.add('North',['ca','mx','us'])
wm.add('Central',['bz','cr','gt','hn'])
wm.add('South',['ar','bo','br','cl'])

wm.render_to_file('usa.svg')
