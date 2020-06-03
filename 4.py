from builtins import float

import cv2 #library opencv
import sys
import math
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow,QDialog,QApplication,QAction
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt
from konvolusi import convolve as conv

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage,self).__init__()
        loadUi('latihan.ui',self)
        self.image=None
        self.loadButton.clicked.connect(self.loadClicked)
        self.saveButton.clicked.connect(self.saveClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightClicked)
        self.actionContrast.triggered.connect(self.contrastClicked)
        self.actionContrastScretch.triggered.connect(self.scretchClicked)
        self.actionNegative.triggered.connect(self.negativeClicked)
        self.actionBiner.triggered.connect(self.binerClicked)
        self.actiongrayHistogram.triggered.connect(self.grayHistogramClicked)
        self.actionRGBHistogram.triggered.connect(self.RGBHistogramClicked)
        self.actionEqualHistogram.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslasi.triggered.connect(self.TranslasiClicked)
        self.actionmin45derajat.triggered.connect(self.minus45Derajat)
        self.actionrot45derajat.triggered.connect(self.rot45Derajat)
        self.actionmin90derajat.triggered.connect(self.minus90Derajat)
        self.actionrot90derajat.triggered.connect(self.rot90Derajat)
        self.actionmin180derajat.triggered.connect(self.minus180Derajat)
        self.actionrot180derajat.triggered.connect(self.rot180Derajat)
        self.actionZoomIn.triggered.connect(self.LinearInterpolationClicked)
        self.actionZoomOut.triggered.connect(self.CubicInterppolationClicked)
        self.actionSkewedImage.triggered.connect(self.skewedSizeClicked)
        self.actionCrop.triggered.connect(self.CropClicked)
        self.actionAritmatika.triggered.connect(self.AritmatikaClicked)
        self.actionBoolean.triggered.connect(self.BooleanClicked)
        self.actionFilter.triggered.connect(self.FilteringClicked)
        self.actionMean.triggered.connect(self.MeanClicked)
        self.actionGaussian.triggered.connect(self.GaussianClicked)
        self.actionSharpening.triggered.connect(self.SharpeningClicked)
        self.actionMedian.triggered.connect(self.MedianClicked)
        self.actionMax.triggered.connect(self.MaxClicked)
        self.actionSobel.triggered.connect(self.SobelClicked)
        self.actionPrewit.triggered.connect(self.PrewitClicked)
        self.actionRobert.triggered.connect(self.RobertClicked)
        self.actionLaplace.triggered.connect(self.LaplaceClicked)
        self.actionLaplaceOfGaussian.triggered.connect(self.LaplaceofGaussianClicked)
        self.actionCanny.triggered.connect(self.CannyClicked)
        self.actionDilasi.triggered.connect(self.Dilasi)
        self.actionErosi.triggered.connect(self.Erosi)
        self.actionOpening.triggered.connect(self.Opening)
        self.actionClosing.triggered.connect(self.Closing)
        self.actionSkeleton.triggered.connect(self.SkeletonClicked)
        self.actionBinary.triggered.connect(self.Binary)
        self.actionBinaryInvers.triggered.connect(self.BinaryInvers)
        self.actionTrunc.triggered.connect(self.Trunc)
        self.actionToZero.triggered.connect(self.ToZero)
        self.actionToZeroInvers.triggered.connect(self.ToZeroInvers)
        self.actionOtsu.triggered.connect(self.otsuClicked)

    @pyqtSlot()
    def grayClicked(self):
        H,W = self.image.shape[:2]
        gray=np.zeros((H,W),np.uint8)
        for i in range (H):
            for j in range (W):
                gray[i,j]=np.clip(0.07*self.image[i,j,0]+0.72*self.image[i,j,1]+0.21*self.image[i,j,2],0,255)
        self.image=gray
        self.displayImage(2)

    @pyqtSlot()
    def loadClicked(self):
        flname,filter=QFileDialog.getOpenFileName(self,'Open File','D:/Foto',"Image Files (*.jpg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    @pyqtSlot()
    def saveClicked(self):
        flname, filter=QFileDialog.getSaveFileName(self,'Save File','D:/',"Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Error')

    def loadImage(self,flname):
        self.image=cv2.imread(flname)
        self.displayImage(1)

    def brightClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        brightness = 50
        h,w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = a + brightness
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                img.itemset((i,j), b)
        self.image = img
        self.displayImage(2)

    def contrastClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h = img.shape[0]
        w =  img.shape[1]
        contrast = 1.6
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255
                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    def scretchClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        max = 255
        min = 0
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                if a > max:
                    max = a
                if a < min:
                    min = a
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = float(a - min) / (max - min) * 255
                img.itemset((i, j), b)

        self.image = img
        self.displayImage(2)

    def negativeClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        max_intensity = 255
        for i in range(h):
            for j in range(w):
                a = img.item(i, j)
                b = max_intensity - a
                img.itemset((i, j), b)
        self.image = img
        self.displayImage(2)

    def binerClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thres = 100
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                if a > thres:
                    a = 255
                elif a < thres:
                    a = 0
                else:
                    a = a
                img.itemset((i, j), a)

        self.image = img
        self.displayImage(2)

    def grayHistogramClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = img
        self.displayImage(2)
        plt.hist(img.ravel(), 255, [0, 255])
        plt.show()

    def RGBHistogramClicked(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        plt.show()

    def EqualHistogramClicked(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def TranslasiClicked(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        t = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.image, t, (w, h))
        self.image = img
        self.displayImage(2)



    def RotasiClicked(self,degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) * (w * cos))
        nH = int((h * cos) * (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h, w))
        self.image = rot_image

    def minus45Derajat(self):
        self.RotasiClicked(-45)
        self.displayImage(2)

    def rot45Derajat(self):
        self.RotasiClicked(45)
        self.displayImage(2)

    def minus90Derajat(self):
        self.RotasiClicked(-90)
        self.displayImage(2)

    def rot90Derajat(self):
        self.RotasiClicked(90)
        self.displayImage(2)

    def minus180Derajat(self):
        self.RotasiClicked(-180)
        self.displayImage(2)

    def rot180Derajat(self):
        self.RotasiClicked(180)
        self.displayImage(2)

    @pyqtSlot()
    def LinearInterpolationClicked(self):
    # make size 3/4 original image size
        cv2.imshow('Original',self.image)
        resize_img=cv2.resize(self.image,None,fx=0.50, fy=0.50)
        self.image=resize_img
        cv2.imshow('',self.image)
        #self.displayImage(2)

    @pyqtSlot()
    def CubicInterppolationClicked(self):
    # double size of original image size/zooming(scaling up)
        cv2.imshow('Original', self.image)
        resize_img=cv2.resize(self.image,None,fx=2,fy=2,interpolation= cv2.INTER_CUBIC)
        self.image = resize_img
        cv2.imshow('',self.image)
        #self.displayImage(2)

    @pyqtSlot()
    def skewedSizeClicked(self):
    # resize image based on exacat dimension
        cv2.imshow('Original', self.image)
        resize_img = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA)
        self.image = resize_img
        cv2.imshow('', self.image)
        # self.displayImage(2)

    def CropClicked(self):
        h, w = self.image.shape[:2]
        # get the strating point of pixel coord(top left)
        start_row, start_col = int(h * .1), int(w * .1)
        # get the ending point coord (botoom right)
        end_row, end_col = int(h * .5), int(w * .5)
        crop = self.image[start_row:end_row, start_col:end_col]
        cv2.imshow('Original', self.image)
        cv2.imshow('Cropped', crop)

    def AritmatikaClicked(self):
        img1 = cv2.imread('2.jpg', 0)
        img2 = cv2.imread('1.jpg', 0)
        add_img = img1 + img2
        subtract = img1 - img2
        subtract2 = img2 - img1
        mul = img1 * img2
        div = img1 / img2
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Add', add_img)
        cv2.imshow('Subtraction', subtract)
        cv2.imshow('Multiply', mul)
        cv2.imshow('Divide', div)

    def BooleanClicked(self):
        img1 = cv2.imread('1.jpg', 1)
        img2 = cv2.imread('2.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and = cv2.bitwise_and(img1, img2)
        op_or = cv2.bitwise_or(img2, img2)
        op_xor = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('And', op_and)
        cv2.imshow('OR', op_or)
        cv2.imshow('XOR', op_xor)

    def FilteringClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = (1.0 / 57) * np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])

        img_out = conv(img, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    def MeanClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = (1.0 / 57) * np.array(
            [[1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9]])

        img_out = conv(img, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    def GaussianClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = (1.0 / 57) * np.array(
            [[0.0029, 0.0131, 0.0215, 0.0131, 0.0029],
             [0.0131, 0.0585, 0.0965, 0.0585, 0.0131],
             [0.0215, 0.0965, 0.1592, 0.0965, 0.0215],
             [0.0131, 0.0585, 0.0965, 0.0585, 0.0131],
             [0.0029, 0.0131, 0.0215, 0.0131, 0.0029]])

        img_out = conv(img, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    def SharpeningClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = (1.0 / 16) * np.array(
            [[1, -2, 1],
             [-2, 5, -2],
             [1, -2, 1]])

        img_out = conv(img, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    @pyqtSlot()
    def MedianClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = self.image.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)

                neighbors.sort()
                median = neighbors[24]
                b = median

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    pyqtSlot()

    @pyqtSlot()
    def MaxClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = self.image.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                b = max
                img_out.itemset((i, j), b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([], plt.yticks([]))
        plt.show()

    pyqtSlot()

    @pyqtSlot()
    def SobelClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        img_x = conv(img, Sx)
        img_y = conv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255

        self.image = img

        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def PrewitClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        Px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        img_x = conv(img, Px)
        img_y = conv(img, Py)

        img_out = np.sqrt((img_x * img_x) + (img_y * img_y))
        img_out = (img_out / np.max(img_out)) * 255

        self.image = img
        self.displayImage(2)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')


        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def RobertClicked(self):
        img = cv2.imread('1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x = np.array([[1, 0], [0, -1]])
        y = np.array([[0, 1], [-1, 0]])
        sx = cv2.filter2D(gray, cv2.CV_64F, x)
        sy = cv2.filter2D(gray, cv2.CV_64F, y)

        hitung = cv2.sqrt((sx * sx) + (sy * sy))
        h, w = hitung.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = hitung.item(i, j)
                if a > 255:
                    a = 255
                elif a < 0:
                    a = 0
                else:
                    a = a

        plt.imshow(hitung, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        print(hitung)
        plt.show()

    def LaplaceClicked(self):
        img = cv2.imread('1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        y = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sx = cv2.filter2D(gray, cv2.CV_64F, x)
        sy = cv2.filter2D(gray, cv2.CV_64F, y)

        hitung = cv2.sqrt((sx * sx) + (sy * sy))
        h, w = hitung.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = hitung.item(i, j)
                if a > 255:
                    a = 255
                elif a < 0:
                    a = 0
                else:
                    a = a

        plt.imshow(hitung, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        print(hitung)
        plt.show()

    def LaplaceofGaussianClicked(self):
        img = cv2.imread('1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_out = cv2.GaussianBlur(gray, (5,5),0)

        x = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        y = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sx = cv2.filter2D(img_out, cv2.CV_64F, x)
        sy = cv2.filter2D(img_out, cv2.CV_64F, y)

        hitung = cv2.sqrt((sx * sx) + (sy * sy))
        h, w = hitung.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = hitung.item(i, j)
                if a > 255:
                    a = 255
                elif a < 0:
                    a = 0
                else:
                    a = a

        plt.imshow(hitung, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        print(hitung)
        plt.show()

    @pyqtSlot()
    def CannyClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gaus = (1.0 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        img_gaus = conv(img, gaus)

        # ubah format gambar
        img_gaus = img_gaus.astype("uint8")
        cv2.imshow("gaus",img_gaus)
        #sobel
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        img_x = conv(img_gaus, Sx)
        img_y = conv(img_gaus, Sy)

        H, W = img.shape[:2]
        img_out = np.zeros((H, W))
        for i in np.arange(H):
            for j in np.arange(W):
                x = img_x.item(i, j)
                y = img_y.item(i, j)
                hasil = np.abs(x) + np.abs(y)
                if (hasil > 255):
                    hasil = 255
                if (hasil < 0):
                    hasil = 0
                else:
                    hasil = hasil
                img_out.itemset((i, j), hasil)

        # hasil gradient
        img_out = img_out.astype("uint8")
        #arah tepi
        theta = np.arctan2(img_y, img_x)
        cv2.imshow("sobel",img_out)
        #non-max supresion
        Z = np.zeros((H, W), dtype=np.int32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]

                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")

        cv2.imshow("NON", img_N)

        # hysteresis Thresholding menentukan tepi lemah dan kuat
        weak = 50
        strong = 100
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak) : #weak
                    b = weak
                    if (a > strong): #strong
                        b = 255
                else:
                    b = 0

                img_N.itemset((i, j), b)

        img_H1 = img_N.astype("uint8")
        cv2.imshow("hysteresis part 1", img_H1)

        # hysteresis Thresholding eliminasi titik tepi lemah jika tidak terhubung dengan tetangga tepi kuat

        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
        (img_H1[i + 1, j] == strong) or
                                  (img_H1[i + 1, j + 1] == strong) or
        (img_H1[i, j - 1] == strong) or
                                  (img_H1[i, j + 1] == strong) or
        (img_H1[i - 1, j - 1] == strong) or
        (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                              img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")
        cv2.imshow("Canny Edge Detection", img_H2)


    @pyqtSlot()
    def Dilasi(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = np.ones((5, 5), np.uint8)
        dilasi = cv2.dilate(thresh, strel, iterations=1)
        cv2.imshow("Dilasi", dilasi)

    def Erosi(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = np.ones((5, 5), np.uint8)
        erosi = cv2.erode(thresh, strel, iterations=1)
        cv2.imshow("Erosi", erosi)

    def Opening(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = np.ones((5, 5), np.uint8)
        Opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, strel, iterations=1)
        cv2.imshow("Opening", Opening)


    def Closing(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = np.ones((5, 5), np.uint8)
        Closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strel, iterations=1)
        cv2.imshow("Closing", Closing)

    @pyqtSlot()
    def SkeletonClicked(self):
        img = cv2.cvtColor(self.image)

        size = np.size(img)
        skeleton = np.zeros(img.shape, np.uint8)

        ret, img = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while (not done):
            erosi = cv2.erode(img, strel)
            temp = cv2.dilate(erosi, strel)

            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = erosi.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        cv2.imshow('Image Asli', img)
        cv2.imshow('Skeletonizing', skeleton)

    @pyqtSlot()
    def Binary(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binary", thresh)

    def BinaryInvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Binary Invers", thresh)

    def Trunc(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
        cv2.imshow("Trunc", thresh)

    def ToZero(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        cv2.imshow("To Zero", thresh)

    def ToZeroInvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
        cv2.imshow("To Zero Invers", thresh)

    @pyqtSlot()
    def otsuClicked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        retval2, otsu_thres = cv2.threshold(img, 125, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean_c = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)
        gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        cv2.imshow("Retval", retval2)
        cv2.imshow("Otsu", otsu_thres)
        cv2.imshow("mean_c", mean_c)
        cv2.imshow("gaus", gaus)



    def displayImage(self, windows=1):
        qformat=QImage.Format_Indexed8

        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)

        #cv membaca image dalam format BGR, PyQt membaca dalam format RGB
        img=img.rgbSwapped()


        #menyimpan gambar hasil load di dalam imgLabel
        if windows==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
        #memposisikan gambar
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

        if windows==2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
        #memposisikan gambar
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.imgLabel2.setScaledContents(True)

#Membuat window enable menampilkan user interface dan kelasnya

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    window=ShowImage()
    window.setWindowTitle('Show Image GUI')
    window.show()
    sys.exit(app.exec_())