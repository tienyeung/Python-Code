# import shutil, os, re

# os.chdir('/home/yeung/图片/')
# rephoto = re.compile(r'.*')
# numphoto=[]
# for file in os.listdir('.'):
#     rename = rephoto.search(file)
#     name = rename.group() 
#     numphoto.append(name)

# const = 0
# for i in numphoto:
#     absdir = os.path.abspath('.')
#     addphoto = os.path.join(absdir,i)
#     const += 1
#     newname='photo'+str(const)+'.jpg'
#     addnew = os.path.join(absdir,newname)
#     print(addnew)
#     shutil.move(addphoto,addnew)
import os

class ImageRename():
    def __init__(self):
        self.path = 'D:/xpu/paper/plate_data'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                os.rename(src, dst)
                print 'converting %s to %s ...' % (src, dst)
                i = i + 1
        print 'total %d to rename & converted %d jpgs' % (total_num, i)

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()


