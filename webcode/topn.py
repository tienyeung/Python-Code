import requests
import sys
import webbrowser
import bs4

if len(sys.argv) > 1:
    res = requests.get('http://google.com/search?q=' + ' '.join(sys.argv[1:]))
    res.raise_for_status()
else:
    res = requests.get("http://google.com/search?q=test")

soup = bs4.BeautifulSoup(res.text)
link = soup.select('.r a')
numopen = min(5, len(link))
for i in range(numopen):
    webbrowser.open('http://google.com'+link[i].get('href'))
