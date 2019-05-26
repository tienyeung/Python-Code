# 爬虫初探

很简单的任务，根据关键词在百度中搜索，返回的网页(第一页)有多个条目，从中选取每个条目的url。

- **requests**打开网页，返回对象

> res = requests.get(url=url(,headers=headers,cookie=cookie))

- **beautifulsoup**解析网页

> soup = Beautifulsoup(res.content,'html.parser')

- 定位tag,**select()**

> tag = soup.select('h3>a')
>
>  or 
>
> tag = soup.find_all('a')#没试过，不能确定a是否只在h3tag下

> 获取的a是所有a属性的列表

- 获取属性值**get()**

> url = tag.get('href')

/TO DO

- 获取前20页的所有条目的url以及title



*反思*

很简单的爬虫任务，我却做了一个小时，- -！惭愧惭愧，归根结底是我对爬虫工具的及其不熟悉以及爬虫原理的不理解，我不不知道如何定位标签，如何获取标签里的属性的值，经此一役，我相信有了更深刻的认识。