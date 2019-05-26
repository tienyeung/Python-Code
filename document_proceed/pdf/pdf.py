# 多个pdf合并选择页面
import PyPDF2，os

# 获取当前目录所有pdf文件名
pdfFilename = []
for filename in os.listdir('.'):
    if filename.endswith('.pdf'):
        pdfFilename.append(filename)
pdfFilename.sort(key=str.lower)
# 创建新pdf
pdfWriter = PyPDF2.PdfFileWriter()
#读取各pdf并添加除封面外的页面到新pdf
for filename in pdfFilename:
    pdfObj = open(filename, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfObj)
    for pageNum in range(1,pdfReader.numPages):
        pageObj=pdfReader.getPage(pageNum)
        pdfWriter.addPage(pageObj)
#save
with open('newpdf.pdf','wb') as f:
    pdfWriter.write(f)
