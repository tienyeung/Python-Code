import sys
from PyQt5.QtWidgets import (QApplication,QLCDNumber,QSlider,QVBoxLayout,QWidget)
from PyQt5.QtCore import Qt

class Exam(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        lcd=QLCDNumber(self)
        sld=QSlider(Qt.Horizontal,self)

        vbox=QVBoxLayout()
        vbox.addWidget(lcd)
        vbox.addWidget(sld)

        self.setLayout(vbox)
        # slot!
        sld.valueChanged.connect(lcd.display)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('signal&slot')
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Exam()
    sys.exit(app.exec_())