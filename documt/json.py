import json,requests,sys

#打印天气预报
if  len(sys.argv)<2:
    print('usage:location')
    sys.exit
location=' '.join(sys,argv[1:])

url = 'http://api.openweathermap.org/data/2.5/forecast/daily?q=%s&cnt3' %(location)
resp=requests.get(url)
resp.raise_for_status

weatherData=json.loads(resp.text)
w=weatherData['list']
print('Current weather in %s:' % (location))
print('tomorrow')
print(w[1]['weather'][0]['main'],'-',w[1]['weather'][0]['description'])