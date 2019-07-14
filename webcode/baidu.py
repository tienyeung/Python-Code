import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}

def search():
    #for i in range(0,21,10):
    url = 'https://www.baidu.com/s?wd=%E6%96%B0%E5%B7%A5%E7%A7%91&pn=0&oq=%E6%96%B0%E5%B7%A5%E7%A7%91&tn=baiduhome_pg&ie=utf-8&usm=1&rsv_idx=2&rsv_pq=cbc6592f00087f88&rsv_t=fae0OYSGYFXejpFkAYyZesdohtsNK7VZHMMZo8QXdvpTL1mhc8clku7L6vsW0in8CjWg'
    res=requests.get(url=url,headers=headers)
    soup = BeautifulSoup(res.content, 'html.parser')
    item_a=soup.select('h3>a')
    for href in item_a:
        url=href.get('href')
        print(url)

if __name__ == '__main__':
    search()