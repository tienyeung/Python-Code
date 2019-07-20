import pyecharts
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib
# myfont = matplotlib.font_manager.FontProperties(fname='../zhaozi.ttf', size=14) # 为了显示中文
# sns.set(font=myfont.get_name())


def get_url(city='kunming'):
    for time in range(201801, 201813):
        url = 'http://lishi.tianqi.com/{}/{}.html'.format(city, time)
        yield url


def get_datas(urls=get_url()):
    cookie = {
        "cityPy": "sanming; cityPy_expire=1551775148; UM_distinctid=16928f54c6d0-08753ecf8a3d56-5d4e211f-1fa400-16928f54c6e445; CNZZDATA1275796416=308465049-1551166484-null%7C1551172369; Hm_lvt_ab6a683aa97a52202eab5b3a9042a8d2=1551170359,1551172902; Hm_lpvt_ab6a683aa97a52202eab5b3a9042a8d2=1551172902"}
    header = {"User-Agent":'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}
    for url in urls:
        html = requests.get(url = url, headers = header, cookies = cookie)
        soup = BeautifulSoup(html.content, 'html.parser')
        date = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(1) > a")
        max_temp = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(2)")
        min_temp = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(3)")
        weather = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(4)")
        wind_direction = soup.select("#tool_site > div.tqtongji2 > ul > li:nth-of-type(5)")
        date = [x.text for x in date]
        max_temp = [x.text for x in max_temp[1:]]
        min_temp = [x.text for x in min_temp[1:]]
        weather = [x.text for x in weather[1:]]
        wind_direction = [x.text for x in wind_direction[1:]]
        yield pd.DataFrame([date, max_temp, min_temp, weather, wind_direction]).T

def get_result():
    result = pd.DataFrame()
    for data in get_datas():  
        result = result.append(data)
    return result


result = get_result()
# print(result)

result.columns = ['日期', '最高温度', '最低温度', '天气', '风向']
result['日期'] = pd.to_datetime(result['日期'])
result["最高温度"] = pd.to_numeric(result['最高温度'])
result["最低温度"] = pd.to_numeric(result['最低温度'])
result["平均温度"] = (result['最高温度'] + result['最低温度'])/2
# print(result)
# 看一下更改后的数据状况
# result.info()

# sns.distplot(result['平均温度'])
# plt.show()

# 按月份统计降雨和没有降雨的天气数量

result['是否降水'] = result['天气'].apply(lambda x: '未降水' if x in ['晴','多云','阴','雾','浮尘','霾','扬沙'] else '降水')
rain = result.groupby([result['日期'].apply(lambda x:x.month),'是否降水'])['是否降水'].count()



month = [str(i)+"月份" for i in range(1,13)]
is_rain = [rain[i]['降水'] if '降水' in rain[i].index else 0 for i in range(1,13)]

no_rain = [rain[i]['未降水'] if '未降水' in rain[i].index else 0  for i in range(1,13)]


line = pyecharts.Line("各月降水天数统计")

line.add(
    "降水天数",
    month,
    is_rain,
    is_fill=False,
    area_opacity=0.7,
    is_stack = True)

line.add(
    "未降水天数",
    month,
    no_rain,
    is_fill=False,
    area_opacity=0.7,
    is_stack = True)
line.render('rain2018.html')



