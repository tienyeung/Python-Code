import requests

url='https://api.github.com/rate_limit'
r=requests.get(url)
r_dict=r.json()

for limit in r_dict:
    print('resourse:',r_dict['resources']['core']['limit'])
    print('search:',r_dict['resources']['search']['limit'])
    print('rate:',r_dict['rate']['limit'])
