import openpyxl
import os
import pprint

# print(os.getcwd())
os.chdir('./document proceed')
print('opening workbook...')
wb = openpyxl.load_workbook('./censuspopdata.xlsx')
# print(type(wb))
sheet = wb.get_sheet_by_name('Population by Census Tract')
countryData = {}

print('reading rows...')
for row in range(2, sheet.max_row+1):
    State = sheet['B'+str(row)].value
    county = sheet['C'+str(row)].value
    pop = sheet['D'+str(row)].value

    # setdefault make sure next key,value for this key exists
    #相当于初始化字典格式，规定好字典嵌套方式
    countryData.setdefault(State, {})
    countryData[State].setdefault(county, {'tracts': 0, 'pop': 0})

    countryData[State][county]['tracts'] += 1
    countryData[State][county]['pop'] += int(pop)

print('writing results...')
with open('census2010.json', 'w') as f:
    f.write('alldata='+pprint.pformat(countryData))
print('Done!')
