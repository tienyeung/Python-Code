import requests
import pygal
from pygal.style import LightColorizedStyle as LCS,LightenStyle as LS

#执行api获取响应
url='https://api.github.com/search/repositories?q=language:python&sort=stars'
r=requests.get(url)
#获取状态为200代表成功
print('status code:',r.status_code)
#将返回来的json存储为字典
response_dict=r.json()
print('Total repositories:',response_dict['total_count'])

#探索仓库信息,items表示仓库的集合
repo_dicts=response_dict['items']
print('repo return:',len(repo_dicts))

#遍历仓库
names,plot_dicts=[],[]
for repo_dict in repo_dicts:
    names.append(repo_dict['name'])

    plot_dict={
        'value':repo_dict['stargazers_count'],
        'label':str(repo_dict['description']),#文件过大，将description转化为str
        'xlink':repo_dict['html_url']
    }#两个标签描述
    plot_dicts.append(plot_dict)


#可视化
my_style=LS('#333366',base_style=LCS)
#样式定制
my_config=pygal.Config()
my_config.x_label_rotation=45
my_config.show_legend=False#隐藏图例
my_config.title_font_size=23
my_config.label_font_size=12
my_config.major_label_font_size=18
my_config.truncate_label=15#缩短较长项目名
my_config.show_y_guides=False#隐藏水平线
my_config.width=1000


chart=pygal.Bar(my_config,style=my_style)
chart.title='Most starred python project in github'
chart.x_labels=names
chart.add('',plot_dicts)

chart.render_to_file('python_repo.svg')
