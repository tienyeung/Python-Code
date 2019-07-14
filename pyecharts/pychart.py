import pandas as pd
from pyecharts import Bar, Line, Overlap, Grid
# 转换数据格式
table = '''
净销售额       净利润     毛利率     现金      总资产     国行     iPhone销量    iPhone销售额
32479.0     4834.0      0.342   11875.0     36171   0           11627.0     6742.0
42905.0     8235.0      0.401   5263.0      47501   769.0       20731.0     13033.0
65225.0     14013.0     0.394   11261.0     75183   2764.0      39989.0     25179.0
108249.0    25922.0     0.405   9815.0      116371  12472.0     72293.0     47057.0
156508.0    41733.0     0.439   10746       176064  22797.0     125046.0    80477.0
170910.0    37037.0     0.376   14259       207000  25946.0     150257.0    91279.0
182795.0    39510.0     0.386   13844       231839  30638.0     169219.0    101991.0
233715.0    53394.0     0.401   21120       290479  56547.0     231218.0    155041.0
215639.0    45687.0     0.391   20484       321686  46349       211884.0    136700.0
229234.0    48351.0     0.385   20289       375319  44764       216756.0    141319.0
'''
table = table.strip().split('\n')
head = table[0].split()
body = table[1:]
data = {}
year = 2008
for line in body:
    data_year = {}
    line = line.split()
    for i in range(len(head)):
        data_year[head[i]] = line[i]
    data[year] = data_year
    year += 1
data = pd.DataFrame(data)

years = [str(i) for i in range(2008, 2018)]
net_sales = data.loc['净销售额'].values
net_income = data.loc['净利润'].values
bar = Bar("盈利能力")
bar.add("净销售额", years, net_sales)
bar.add("净利润", years, net_income, bar_category_gap=25, yaxis_name='百万美元', yaxis_name_gap=60)
gross = data.loc['毛利率'].values
line = Line()
line.add("毛利率", years, gross, line_width=3)
ol = Overlap()
ol.add(bar)
ol.add(line, is_add_yaxis=True, yaxis_index=1)
ol.render()

assets = data.loc['总资产'].values
cash = data.loc['现金'].values
bar = Bar("财务状况")
bar.add("总资产", years, assets)
bar.add("现金", years, cash, bar_category_gap=25, yaxis_name='百万美元', yaxis_name_gap=60)
bar.render()





ip_sales = data.loc['iPhone销售额'].values
ip_unit = data.loc['iPhone销量'].values
bar = Bar("iPhone销售状况")
bar.add("iPhone销售额", years, ip_sales)
bar2 = Bar()
bar2.add("iPhone销量", years, ip_unit, bar_category_gap=25)
percent = ip_sales.astype('float') / net_sales.astype('float')
line = Line()
line.add("收入占比", years, percent, line_width=3, yaxis_margin=60, yaxis_pos='left')
ol = Overlap()
ol.add(bar)
ol.add(bar2, is_add_yaxis=True, yaxis_index=1)
ol.add(line, is_add_yaxis=True, yaxis_index=2)
grid = Grid()
grid.add(ol, grid_left="15%")
grid.render()



cn_sales = data.loc['国行'].values
bar = Bar("国行销售状况")
bar.add("国行", years, cn_sales)
percent = cn_sales.astype('float') / net_sales.astype('float')
line = Line()
line.add("国行占比", years, percent, line_width=3)
ol = Overlap()
ol.add(bar)
ol.add(line, is_add_yaxis=True, yaxis_index=1)
ol.render()
