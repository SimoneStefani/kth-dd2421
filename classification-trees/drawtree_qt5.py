#!/usr/bin/env python

import dtree
import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow


class MyPainting(QWidget):
    def __init__(self, parent, tree):
        QWidget.__init__(self)
        self.tree = tree
        self.xsize = 600
        self.ysize = 300

    def sizeHint(self):
        return QtCore.QSize(self.xsize, self.ysize)

    def resizeEvent(self, ev):
        size = ev.size()
        self.xsize = size.width()
        self.ysize = size.height()

    def xscale(self, x1, x2):
        return self.xsize/2.0 + (x1-x2)*(self.xsize - 10) * 0.9

    def yscale(self, y):
        return 10 + (y/12.0)*(self.ysize - 20)

    def paintEvent(self, ev):
        p = QtGui.QPainter()
        p.begin(self)
        p.setPen (QtGui.QPen(QtGui.QColor(0,0,0), 1))
        draw(p, self.tree, 10, 10)
        p.end()


def draw(p, t, x, y):
    if isinstance(t, dtree.TreeLeaf):
        p.drawText(x-3, y+15, 'T' if t.cvalue else 'F')
        return x, x+20
    xx = x
    anchors = []
    for b in t.branches:
        mid, xx = draw(p, t.branches[b], xx, y+70)
        p.drawText(mid-3, y+68, str(b))
        anchors.append(mid)
    newMid = (x+xx)/2
    p.drawText(newMid-7, y+15, t.attribute.name)
    p.drawEllipse(newMid-15, y, 30, 20)
    for m in anchors:
        p.drawLine(newMid, y+20, m, y+70)
    return newMid, xx+10


class MyMainWindow( QMainWindow ):
    def __init__(self, tree):
        QMainWindow.__init__( self )
        paint = MyPainting(self, tree)
        self.setCentralWidget(paint)
        self.show()


def drawTree(tree):
    application = QApplication(sys.argv)
    win = MyMainWindow(tree)  

    win.show()
    sys.exit(application.exec_())










