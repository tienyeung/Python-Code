import requests,bs4,os,threading

os.makedirs('xkcd',exist_ok=True)

def downloadXkcd(startComic,endComic):
    #获取url
    for urlNum in range(startComic,endComic):
        print('downloading page http://xkcd.com/%s' %(urlNum))
        res=requests.get('http://xkcd.com/%s' %(urlNum))
        res.raise_for_status
    
    #获取html
    soup=bs4.BeautifulSoup(res.text)

    #找到图片的src
    #'select'返回Tag的一个列表
    #'get'返回Tag的属性的值
    comicElem=soup.select('#comic img')
    if comicElem==[]:
        print('couldn't find the image)
    else:
        comicUrl=comicElem[0].get('src')
    res=requests.get(comicUrl)
    res.raise_for_status

    #iter_content返回一段内容，上限100k字节
    with open(os.path.join('xkcd',os.path.basename(comicUrl)),'wb') as f:
        for chunk in res.iter_content(100000):
            f.write(chunk)
    
    #启动线程
    downloadThreads=[]
    for i in range(0,1400,100):
        downloadThread=threading.Thread(target=downloadXkcd,args=(i,i+99))
        downloadThreads.append(downloadThread)
        downloadThread.start()

    #所有线程结束后才调用主程序
    for downloadThread in downloadThreads:
        downloadThread.join()
    print('Done!')        

