# Goose 是一个文章内容提取器，可以从任意资讯文章类的网页中提取文章主体，并提取标题、标签、摘要、图片、视频等信息
from goose3 import Goose
from goose3.text import StopWordsChinese
from bs4 import BeautifulSoup


g = Goose({'stopwords_class': StopWordsChinese})
urls = [
    'https://www.ifanr.com/',
    'https://www.leiphone.com/',
    'http://www.donews.com/'
]
url_articles = []
for url in urls:
    page = g.extract(url=url)
    soup = BeautifulSoup(page.raw_html, 'lxml')  # raw_html：原始 HTML 文本
    links = soup.find_all('a')
    for l in links:
        link = l.get('href')
        if link and link.startswith('http') and any(c.isdigit() for c in link if c) and link not in url_articles:
            url_articles.append(link)
            print(link)

for url in url_articles:
    try:
        article = g.extract(url=url)
        content = article.cleaned_text
        if len(content) > 200:
            title = article.title
            print(title)
            with open('/home/yeung/PycharmProjects/untitled/goose/' + title + '.txt', 'w') as f:
                f.write(content)
    except:
        pass
