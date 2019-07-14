import shutil, os, re

os.chdir('/home/yeung/图片/')
rephoto = re.compile(r'.*')
numphoto=[]
for file in os.listdir('.'):
    rename = rephoto.search(file)
    name = rename.group() 
    numphoto.append(name)

const = 0
for i in numphoto:
    absdir = os.path.abspath('.')
    addphoto = os.path.join(absdir,i)
    const += 1
    newname='photo'+str(const)+'.jpg'
    addnew = os.path.join(absdir,newname)
    print(addnew)
    shutil.move(addphoto,addnew)

