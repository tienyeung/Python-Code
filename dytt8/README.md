# scrapy

> **scrapy**是一个强大的高level的爬虫框架，内部封装了许多函数可轻易构建高性能爬虫。

应用scrapy框架制作简易的爬虫，通过爬取电影天堂，获取各电影的下载地址并存入进mongodb

## 技术栈：

- scrapy的认知
- mongodb的配置
- xpath初步了解

## 流程：

1. **新建项目**

> scrapy startproject www_dytt8_net

通过新建项目可获取相关文件目录

> ```
> mySpider/
>     scrapy.cfg
>     mySpider/
>         __init__.py
>         items.py
>         pipelines.py
>         settings.py
>         spiders/
>             __init__.py
>             ...
> ```

这些文件分别是:

- scrapy.cfg: 项目的配置文件。
- mySpider/: 项目的Python模块，将会从这里引用代码。
- mySpider/items.py: 项目的目标文件。
- mySpider/pipelines.py: 项目的管道文件。
- mySpider/settings.py: 项目的设置文件。
- mySpider/spiders/: 存储爬虫代码目录。



2. **创建crawl模板的爬虫demo**

进入到spiders目录，创建爬虫模板

> scrapy genspider -t crawl dytt8 www.dytt8.net



3. **配置跟踪规则（rule）**

[scrapy rule语法](https://scrapy.readthedocs.io/en/latest/topics/link-extractors.html#module-scrapy.linkextractors.lxmlhtml)

只需要跟踪首页链接中，带有index的页面，进入到内页后，我们继续跟踪下一页即可遍历所有电影页面。

> ```
> Rule(LinkExtractor(deny=r'.*game.*', allow='.*/index\.html'))
> ```

正则表达式表示排除带有game的字段，跟踪带有index的html

然后导航页点击下一页

> ```
> Rule(LinkExtractor(restrict_xpaths=u'//a[text()="下一页"]'))
> ```

`restrict_xpaths` 支持xpath语法，提取标签页为下一页的所有a标签内的链接

之后提取文章页链接，交由解析函数处理

> ```
> Rule(LinkExtractor(allow=r'.*/\d+/\d+\.html', deny=r".*game.*"), callback='parse_item', follow=True)
> ```

**callback**表示交由*parse_item*函数处理，**follow**为对此页面进一步跟踪



4. **提取所需信息**

在items.py内定义我们需要提取的字段

> ```
> import scrapyclass WwwDytt8NetItem(scrapy.Item):
>     # define the fields for your item here like:
>     # name = scrapy.Field()
>     title = scrapy.Field()
>     publish_time = scrapy.Field()
>     images = scrapy.Field()
>     download_links = scrapy.Field()
>     contents = scrapy.Field()
> ```

接着将spiders/dytt8.py中的回调函数中的提取规则写出来



5. **启动脚本**

> scrapy crawl dytt8



## MongoDB

settings.py中，配置如下中间键

> ```
> ITEM_PIPELINES = {
>    # 'www_dytt8_net.pipelines.WwwDytt8NetPipeline': 300,
>    'scrapy_mongodb.MongoDBPipeline': 300,
> }
> ```

然后写入mongodb配置即可

> ```
> MONGODB_URI = 'mongodb://localhost:27017'
> MONGODB_DATABASE = 'spider_world'
> MONGODB_COLLECTION = 'dytt8'
> ```

