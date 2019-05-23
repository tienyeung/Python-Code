import sys
from PyQt5.QtWidgets import (QApplication,QWidget,QLabel,QComboBox)

class Exam(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl=QLabel('Arch',self)

        combo=QComboBox(self)
        combo.addItem('Arch')
        combo.addItem('Manjaro')
        combo.addItem('Elementary')
        combo.addItem('CentOS')

        combo.move(50,50)
        self.lbl.move(50,150)

        combo.activated[str].connect(self.onActivated)



        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QComboBox')
        self.show()
    
    def onActivated(self,text):
        self.lbl.setText(text)
        self.lbl.adjustSize

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Exam()
    sys.exit(app.exec_())
