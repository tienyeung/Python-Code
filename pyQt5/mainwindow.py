import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QAction, qApp, QMenu,QTextEdit)
from PyQt5.QtGui import QIcon


class Exam(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        exitAct=QAction(QIcon('./icon.jpg'),'&Exit',self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit')
        exitAct.triggered.connect(qApp.quit)

        self.statusBar()

        #设置文本框
        textEdit=QTextEdit()
        self.setCentralWidget(textEdit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)

        # 工具栏
        toolBar=self.addToolBar('Exit')
        toolBar.addAction(exitAct)
        

        # 子菜单
        impMenu = QMenu('Import', self)
        impAct = QAction('Improt mail', self)
        # 子菜单里添加动作
        impMenu.addAction(impAct)

        # 主菜单添加动作
        newAct = QAction('New', self)
        fileMenu.addAction(newAct)

        # 主菜单添加子菜单
        fileMenu.addMenu(impMenu)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Menu')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Exam()
    sys.exit(app.exec_())
