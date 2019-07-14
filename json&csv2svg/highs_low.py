import csv
from datetime import datetime
from matplotlib import pyplot as plt

filename='/Users/apple/Desktop/download_file/sitka_weather_2014.csv'
with open(filename) as f:
    reader=csv.reader(f)
    header_row=next(reader)

    #for index,column_header in enumerate(header_row):#获取索引及值
        #print(index,column_header)

    dates,highs,lows=[],[],[]
    for row in reader:
        date_time=datetime.strptime(row[0],'%Y-%m-%d')
        dates.append(date_time)
        high=int(row[1])
        highs.append(high)
        low=int(row[3])
        lows.append(low)

    fig=plt.figure(dpi=128,figsize=(10,6))

    plt.plot(dates,highs,c='red',linewidth=2)
    plt.plot(dates,lows,c='green',linewidth=2)
    plt.fill_between(dates,highs,lows,facecolor='yellow',alpha=0.5)

    plt.title('Daily high and low temperatures-2014',fontsize=24)
    plt.xlabel('',fontsize=16)
    fig.autofmt_xdate()
    plt.ylabel('Temperature',fontsize=16)
    plt.tick_params(axis='both',which='major',labelsize=16)

    plt.savefig('high_low_temper.png')
    plt.show()
