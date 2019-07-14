import csv
import os

#删除每个csv的表头
os.makedirs('headerRomove'，exist_ok=True)

for csvFilename in os.listdir('.'):
    if not csvFilename endwith('.csv'):
        continue
    print('Removing header from %s ...',%csvFilename)
    csvRows=[]
    csvFileObj=open(csvFilename)
    csvReader=csv.reader(csvFileObj)
    for row in csvReader:
        if csvReader.line_num==1:
            continue
        csvRows.append(row)
    csvFileObj.close()
    
    #write out the CSV file
    csvFileObj=open(os.path.join('headerRomoved',csvFilename),'w',newline='')
    csvWriter=csv.writer(csvFileObj)
    for row in csvRows:
        csvWriter.writerow(row)
    csvFileObj.close() 