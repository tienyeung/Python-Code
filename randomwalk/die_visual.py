import pygal
from die import Die

die_1=Die()
die_2=Die()
results=[]
for roll_num in range(1000):
    result=die_1.roll()+die_2.roll()
    results.append(result)

#分析结果
max_result=die_1.num_sides+die_2.num_sides
frequencies=[]
for value in range(2,max_result+1):
    frequency=results.count(value)
    frequencies.append(frequency)

#结果可视化
hist=pygal.Bar()
hist.title='Result of rolling two D6 1000 times'
hist.x_labels=list(range(2,13))
hist.x_title='Result'
hist.y_title='Frequence of Result'

hist.add('D6+D6',frequencies)#纵坐标传递的标签和值
hist.render_to_file(filename='die.svg')
