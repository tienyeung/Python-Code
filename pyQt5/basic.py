import sys
from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QDesktopWidget)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QCoreApplication


class Exam(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        #self.setToolTip('This is a <b>QWidget</b> widget')

        # 创建button
        btn = QPushButton('Quit', self)
        btn.setToolTip('<b>Quit</b>!')
        btn.clicked.connect(QCoreApplication.instance().quit)
        btn.resize(btn.sizeHint())
        btn.move(50, 50)

        # icon
        self.setWindowIcon(QIcon('icon.jpg'))

        # 窗口画布
        self.setGeometry(300, 300, 300, 200)
        self.center()
        self.setWindowTitle('ToolTips')
        self.show()


    def closeEvent(self, event):
        '''关闭进一步提醒'''
        reply = QMessageBox.question(
            self, 'Warn', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self):
        '''窗口居中'''
        qr = self.frameGeometry()  # 得到窗口大小
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕分辨率，定位中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Exam()
    sys.exit(app.exec_())
