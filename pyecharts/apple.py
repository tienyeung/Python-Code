import requests
import urllib.request
import pandas as pd

# 获取年报文件
url = 'http://investor.apple.com/feed/SECFiling.svc/GetEdgarFilingList?apiKey=BF185719B0464B3CB809D23926182246&exchange=CIK&symbol=0000320193&formGroupIdList=1%2C4&excludeNoDocuments=true&pageSize=-1&pageNumber=0&tagList=&includeTags=true&year=-1&excludeSelection=1'
rsp = requests.get(url)
data = rsp.json()  # 解析url json

# 下载
for year_data in data['GetEdgarFilingListResult']:
    year = year_data['FilingDate'].split()[0].split('/')[-1]
    for doc in year_data['DocumentList']:
        if doc['DocumentType'] == 'XLS':
            url_xls: object = doc['Url']
            break
    print(year, url_xls)
    urllib.request.urlretrieve(url_xls, str(year) + '.xls')

# 从文件中搜索相关信息
for y in range(2008, 2018):
    print('\n-------------------', y)
    ex = pd.ExcelFile('/home/yeung/PycharmProjects/machine_reading/%d.xls' % y)
    sheets = pd.read_excel(ex, None)
    for sheet in sheets:
        s = sheets[sheet]
        for index, row in s.iterrows():
            line = str(row.values)
            #             if 'iphone' in line.lower() and '.' in line:
            #             if 'total net sales' in line.lower() and '.' in line:
            #             if 'net income' in line.lower() and '.' in line:
            #             if 'gross margin percentage' in line.lower() and '.' in line:
            #             if 'cash equivalents' in line.lower():
            #             if 'total assets' in line.lower():
            if 'china' in line.lower():
                print(sheet)
                print(line, '\n')
    ex.close()


