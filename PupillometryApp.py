import os
import sys
import cv2
import time
import imutils
import serial
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5 import QtGui 
from PyQt5 import uic
from PyQt5 import QtCore
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import xml.etree.ElementTree as ET

GLOBAL_STATE = 0
scatterMCV = 0
scatterMCA = 0

## Threading
class TaskThread(QtCore.QThread):
    notifyProgressBar = QtCore.pyqtSignal(int)
    notifyConnect = QtCore.pyqtSignal(str)
    valueConnect = QtCore.pyqtSignal(str)
    def __init__(self, varCOM, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.valueCOM = varCOM

    def run(self):
        count = 0
        try:
            self.port = serial.Serial(self.valueCOM, 9600)
            for i in range(50):
                count += 2
                self.notifyProgressBar.emit(count)
                if count == 50:
                    self.port.write(str.encode('5'))
                if count == 80:
                    if self.port.read_all() == b'Y':
                        self.notifyConnect.emit("Connected")
                        self.port.close()
                        self.valueConnect.emit(self.valueCOM)
                    else:
                        self.notifyConnect.emit("Not Connected")
                        self.port.close()
                time.sleep(0.1)

        except:
            self.port = 0
            self.notifyProgressBar.emit(50)
            time.sleep(1)
            self.notifyProgressBar.emit(100)
            self.notifyConnect.emit("Not Connected")

class ThreadCameraGet(QtCore.QThread):
    def __init__(self, indexCamera):
        QtCore.QThread.__init__(self)
        self.stopped = False
        self.stream = cv2.VideoCapture(indexCamera, cv2.CAP_DSHOW)
        self.stream.set(3, 640)
        self.stream.set(4, 480)
        (self.grabbed, self.frame) = self.stream.read()
    
    def run(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()       
        self.stream.release()
        self.quit()

    def stop(self):
        self.stopped = True

class ThreadShowFrame(QtCore.QThread):
    showImage = QtCore.pyqtSignal(QtGui.QImage)
    def __init__(self, VideoFrame, widthVid, heightVid):
        QtCore.QThread.__init__(self)
        self.stopped = False
        self.frame = VideoFrame
        self.widthVideo = widthVid
        self.heightVideo = heightVid
        self.image = imutils.resize(self.frame, width=self.widthVideo, height=self.heightVideo)
        self.image = QtGui.QImage(self.image, self.image.shape[1],self.image.shape[0],self.image.strides[0], QtGui.QImage.Format_BGR888)

    def run(self):
        while not self.stopped:
            self.image = imutils.resize(self.frame, width=self.widthVideo, height=self.heightVideo)
            self.image = QtGui.QImage(self.image, self.image.shape[1],self.image.shape[0],self.image.strides[0], QtGui.QImage.Format_BGR888) 
        self.quit()

    def stop(self):
        self.stopped = True

## Canvas Matplotlib
class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi = 70):
        global GLOBAL_Figure
        self.fig = Figure(dpi = dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas,self).__init__(self.fig)
        self.fig.subplots_adjust(0.18, 0.15, 0.95, 0.95)

## Main App
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("PupillometryApp.ui", self)

        # Setting Tampilan Utama
        self.main_page.setCurrentWidget(self.home_page)

        # Tampilan Halaman Home
        self.btn_home.clicked.connect(lambda: self.main_page.setCurrentWidget(self.home_page))
        self.btn_home_icon.clicked.connect(lambda: self.main_page.setCurrentWidget(self.home_page))
        self.btn_home.clicked.connect(self.changeStyleBtn)
        self.btn_home_icon.clicked.connect(self.changeStyleBtn)

        # Tampilan Halaman Analytic
        self.btn_analytics.clicked.connect(lambda: self.main_page.setCurrentWidget(self.analytics_page))
        self.btn_analytics_icon.clicked.connect(lambda: self.main_page.setCurrentWidget(self.analytics_page))
        self.btn_analytics.clicked.connect(self.changeStyleBtn_1)
        self.btn_analytics_icon.clicked.connect(self.changeStyleBtn_1)

        # Tampilan Halaman Setting
        self.btn_setting.clicked.connect(lambda: self.main_page.setCurrentWidget(self.setting_page))
        self.btn_setting_icon.clicked.connect(lambda: self.main_page.setCurrentWidget(self.setting_page))
        self.btn_setting.clicked.connect(self.changeStyleBtn_2)
        self.btn_setting_icon.clicked.connect(self.changeStyleBtn_2)

        # Button ON/OFF Camera
        self.btn_on_off_camera.clicked.connect(self.CameraButton)
        self.btn_on_off_led.clicked.connect(self.LedButton)

        # Button Start/Stop Pupils Process
        self.btn_start_process.clicked.connect(self.startPupilsProcess)
        self.btn_stop_process.clicked.connect(self.stopPupilsProcess)

        # Button Strat/Stop Video Pupils Process
        self.btn_video_process.clicked.connect(self.startVideoProcess)
        self.btn_stop_video_process.clicked.connect(self.stopVideoProcess)

        # Button Data Analytic
        self.select_MCV.clicked.connect(self.processMCV)
        self.select_MCA.clicked.connect(self.processMCA)
        self.btn_save_data.clicked.connect(self.saveData)
        self.btn_clear.clicked.connect(self.clearData)
        self.btn_update_data.clicked.connect(self.updateData)

        # Button Setting Window
        self.btn_save_setting.clicked.connect(self.saveSetting)
        self.btn_default_setting.clicked.connect(self.defaultSetting)
        ###### Device Connection ######
        self.btn_connecting.clicked.connect(self.deviceConnecting)

        # Set Grafik
        self.canv = MatplotlibCanvas(self)
        self.canv1 = MatplotlibCanvas(self)
        self.canv2 = MatplotlibCanvas(self)

        # Get Setting 
        if os.path.exists("setting.xml") == True:
            dataSetting = ET.parse('setting.xml')
            ###### Timer LED ########
            self.duration = dataSetting.find('Duration').text
            self.duration = int(self.duration)
            self.whiteLedON = dataSetting.find('WhiteLedON').text
            self.whiteLedON = int(self.whiteLedON)
            self.whiteLedOFF = dataSetting.find('WhiteLedOFF').text
            self.whiteLedOFF = int(self.whiteLedOFF)

            self.set_duration.setValue(self.duration)
            self.set_white_led_on.setMaximum(self.duration)
            self.set_white_led_on.setValue(self.whiteLedON)
            self.set_white_led_off.setMaximum(self.duration)
            self.set_white_led_off.setValue(self.whiteLedOFF)

            ##### Connection Device #####
            self.valueCOM = dataSetting.find('PortDevice').text

            self.set_port_device.setCurrentText(self.valueCOM)

            ##### Setting Plot Graph ######
            self.avgDiameter = dataSetting.find('AvgDiameter').text
            self.avgDiameter = int(self.avgDiameter)
            self.avgVelocity = dataSetting.find('AvgVelocity').text
            self.avgVelocity = int(self.avgVelocity)
            self.avgAcceleration = dataSetting.find('AvgAcceleration').text
            self.avgAcceleration = int(self.avgAcceleration)

            self.set_avg_diameter.setValue(self.avgDiameter)
            self.set_avg_velocity.setValue(self.avgVelocity)
            self.set_avg_acceleration.setValue(self.avgAcceleration)

            #### Setting Video #####
            self.fps_in_recording_text = dataSetting.find('FPSinRecording').text
            self.fps_in_recording = int(self.fps_in_recording_text)
            self.fps_in_display_text = dataSetting.find('FPSinDisplay').text
            self.fps_in_display = int(self.fps_in_display_text)

            self.set_fps_recording.setCurrentText(self.fps_in_recording_text)
            self.set_fps_display.setCurrentText(self.fps_in_display_text)

            actualFrameCalc = self.duration*self.fps_in_recording
            self.actual_frame.setText(str(actualFrameCalc))
            calculation_display1 = 1/self.fps_in_display
            self.actualFPSDisplay = calculation_display1/(1/self.fps_in_recording)
            self.actualFPSDisplay = int(self.actualFPSDisplay)

            #### Set Superscript ####
            self.label_mca.setText("mm/s<sup>2</sup>")
            self.label_aca.setText("mm/s<sup>2</sup>")

        if os.path.exists("setting.xml") == False:
            self.defaultSetting()
            self.saveSetting()

        # Default Setup
        if os.path.exists("TemporaryData") != True:
            os.makedirs("TemporaryData")
        if os.path.exists("MeasurementData") != True:
            os.makedirs("MeasurementData")
            
        self.keyCamera = False
        self.keyPupilsProcess = False
        self.keyVideoProcess = False
        self.filenameVideo = 'TemporaryData/TemporaryVideo.avi'
        self.port = 0
        self.otomaticGrapMCV = 0
        self.otomaticGrapMCA = 0
        self.getSelectGrap = 0
        self.hideButtonAnalytics()

        # Koneksi Device
        self.deviceConnecting()
        
        for i in range(0, 4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            test, frame = cap.read()
            if test == True:
                self.device_status_camera.setText("Connected")
                self.set_camera.setCurrentIndex(i)
                break
            if test == False:
                self.device_status_camera.setText("Not Connected")
                self.set_camera.setCurrentIndex(0)

## Tampilan Setting Title Bar Window
        # Pindah Window
        def moveWindow(event):  
            # Restore Sebelum Pindah
            if self.returnStatus() == 1:
                self.maximize_restore()

            # Jika Klik Kiri Untuk Pindah Window
            if event.buttons() == QtCore.Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # Set Title Bar
        self.header.mouseMoveEvent = moveWindow

        # Set Definitions
        self.uiDefinitions()

    # App Events
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    # Fungsi Maximize Restore
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE

        # Jika Tidak Maximize
        if status == 0:
            self.showMaximized()

            GLOBAL_STATE = 1

            # Jika Maximize Hapus Margin & Border Radius
            self.central_layout.setContentsMargins(0, 0, 0, 0)
            self.main_frame.setStyleSheet("border-radius: 0px;")
            self.header.setStyleSheet("border-radius: 0px; background-color: rgb(47, 46, 52);")
            self.footer.setStyleSheet("border-radius: 0px; background-color: rgb(47, 46, 52);")
            self.btn_maximize.setToolTip("Restore")
            self.btn_maximize.setIcon(QtGui.QIcon("Logo/restore.png"))

        else:
            GLOBAL_STATE = 0
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.central_layout.setContentsMargins(10, 10, 10, 10)
            self.main_frame.setStyleSheet("border-radius: 10px;")
            self.header.setStyleSheet("border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-right-radius: 0px; border-bottom-left-radius: 0px; background-color: rgb(47, 46, 52);")
            self.footer.setStyleSheet("border-top-left-radius: 0px; border-top-right-radius: 0px; border-bottom-right-radius: 10px; border-bottom-left-radius: 10px; background-color: rgb(47, 46, 52);")
            self.btn_maximize.setToolTip("Maximize")
            self.btn_maximize.setIcon(QtGui.QIcon("Logo/maximize.png"))

    # UI Definitions
    def uiDefinitions(self):

        # Menghapus Title Bar
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Set Dropshadow Window
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QtGui.QColor(0, 0, 0, 100))

        # Memberikan Dropshadow Pada Frame
        self.main_frame.setGraphicsEffect(self.shadow)

        # Maximize/Restore
        self.btn_maximize.clicked.connect(self.maximize_restore)

        # Minimize
        self.btn_minimize.clicked.connect(lambda: self.showMinimized())

        # Close
        self.btn_close.clicked.connect(self.closeWindow)

    def closeWindow(self):
        self.keyCamera = False
        self.keyPupilsProcess = False
        self.keyVideoProcess = False
        self.close()

    # Mengembalikan Status Jika Window Pada Keadaan Maximize
    def returnStatus(self):
        return GLOBAL_STATE

## Design Side Menu
    # Memberikan Perubahan Tampilan Pada Side Menu Jika Menu Tersebut Terbuka
    def changeStyleBtn(self):
        self.homeButton.setStyleSheet("background-color: rgb(57, 57, 63);")
        self.analyticsButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")
        self.settingButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")

    def changeStyleBtn_1(self):
        self.homeButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")
        self.settingButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")
        self.analyticsButton.setStyleSheet("background-color: rgb(57, 57, 63);")

    def changeStyleBtn_2(self):
        self.homeButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")
        self.analyticsButton.setStyleSheet("QFrame::hover{ background-color: rgb(57, 57, 63);}")
        self.settingButton.setStyleSheet("background-color: rgb(57, 57, 63);")

## Hide Button Analytics 
    def hideButtonAnalytics(self):
        self.select_MCV.setEnabled(False)
        self.select_MCV.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66,66,66);border-radius:10px;}")
        self.select_MCA.setEnabled(False)
        self.select_MCA.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66,66,66);border-radius:10px;}")
        self.btn_update_data.setEnabled(False)
        self.btn_update_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66,66,66);border-radius:10px;}")
        self.btn_save_data.setEnabled(False)
        self.btn_save_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66,66,66);border-radius:10px;}")
        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66,66,66);border-radius:10px;}")
        self.int_nama.setEnabled(False)
        self.int_nik.setEnabled(False)
        self.int_tempat_lahir.setEnabled(False)
        self.int_tanggal_lahir.setEnabled(False)
        self.int_jenis_kelamin.setEnabled(False)
        self.int_pekerjaan.setEnabled(False)
        self.int_pendidikan.setEnabled(False)
        self.int_agama.setEnabled(False)
        self.textEdit.setEnabled(False)
        self.hasil_d.setEnabled(False)
        self.hasil_tl.setEnabled(False)
        self.hasil_tc.setEnabled(False)
        self.hasil_mcv.setEnabled(False)
        self.hasil_acv.setEnabled(False)
        self.hasil_mca.setEnabled(False)
        self.hasil_aca.setEnabled(False)

## Open Hide Button Analtics
    def openHideButtonAnalytics(self):
        self.select_MCV.setEnabled(True)
        self.select_MCV.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        self.select_MCA.setEnabled(True)
        self.select_MCA.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        self.btn_clear.setEnabled(True)
        self.btn_clear.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        self.int_nama.setEnabled(True)
        self.int_nik.setEnabled(True)
        self.int_tempat_lahir.setEnabled(True)
        self.int_tanggal_lahir.setEnabled(True)
        self.int_jenis_kelamin.setEnabled(True)
        self.int_pekerjaan.setEnabled(True)
        self.int_pendidikan.setEnabled(True)
        self.int_agama.setEnabled(True)
        self.textEdit.setEnabled(True)
        self.hasil_d.setEnabled(True)
        self.hasil_tl.setEnabled(True)
        self.hasil_tc.setEnabled(True)
        self.hasil_mcv.setEnabled(True)
        self.hasil_acv.setEnabled(True)
        self.hasil_mca.setEnabled(True)
        self.hasil_aca.setEnabled(True)

## Button Camera On/Off
    def CameraButton(self):
        if self.keyCamera == False and self.keyPupilsProcess == False and self.keyVideoProcess == False and self.btn_on_off_camera.text() == "Start Cam":
            self.btn_on_off_camera.setText("Stop Cam")
            self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);}")
            self.keyCamera = True
            self.hideSetting()
            self.videoFromWebcam()

        if self.keyCamera == True and self.btn_on_off_camera.text() == "Stop Cam":
            self.btn_on_off_camera.setText("Start Cam")
            self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
            self.keyCamera = False
            self.set_camera.setEnabled(True)
            self.openSetting()

    def LedButton(self):
        if self.btn_on_off_led.text() == "Led On" and self.keyPupilsProcess == False:
            try:
                self.port.write(str.encode('2'))
                self.btn_on_off_led.setText("Led Off")
                self.btn_on_off_led.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);}")
            except:
                self.device_status_control.setText("Not Connected")
                self.device_status_connection.setText("Not Connected")
                if self.port != 0:
                    self.port.close()
                    self.port = 0
                pass     

        elif self.btn_on_off_led.text() == "Led Off":
            try:
                self.port.write(str.encode('3'))
                self.btn_on_off_led.setText("Led On")
                self.btn_on_off_led.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
            except:
                self.btn_on_off_led.setText("Led On")
                self.btn_on_off_led.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.device_status_control.setText("Not Connected")
                self.device_status_connection.setText("Not Connected")
                if self.port != 0:
                    self.port.close()
                    self.port = 0
                pass 

## Button Pupils Process
    def startPupilsProcess(self):
        if self.keyCamera == True:
            self.clearData()
            self.total_frame.setText("0")
            self.keyPupilsProcess = True
            self.keyCamera = False

    def stopPupilsProcess(self):
        if self.keyPupilsProcess == True:
            self.keyPupilsProcess = False
        
## Button Video Process
    def startVideoProcess(self):
        if self.keyCamera == False and self.keyPupilsProcess == False and self.keyVideoProcess == False:
            self.clearData()
            self.total_frame.setText("0")
            self.keyVideoProcess = True
            self.getvideoProcess()

    def stopVideoProcess(self):
        if self.keyVideoProcess == True:
            self.keyVideoProcess = False

## Display Video From Camera Webcam
    def videoFromWebcam(self):
        vid = cv2.VideoCapture(self.set_camera.currentIndex(), cv2.CAP_DSHOW)
        vid.set(3, 640)
        vid.set(4, 480)        
        while(True):
            img, self.image = vid.read()
            if img == False:
                self.keyCamera = False
                self.btn_on_off_camera.setText("Start Cam")
                self.device_status_camera.setText("Not Connected")
                self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.set_camera.setEnabled(True)
                self.openSetting()
                break

            self.set_camera.setEnabled(False)
            self.device_status_camera.setText("Connected")
            self.image = cv2.flip(self.image, +1)
            self.frameData = np.hstack([self.image])
            self.setPhoto()

            key = cv2.waitKey(0) & 0xFF
            if self.keyCamera == False:
                if self.btn_on_off_led.text() == "Led Off":
                    try:
                        self.port.write(str.encode('3'))
                        self.btn_on_off_led.setText("Led On")
                        self.btn_on_off_led.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                    except:
                        self.btn_on_off_led.setText("Led On")
                        self.btn_on_off_led.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                        self.device_status_control.setText("Not Connected")
                        self.device_status_connection.setText("Not Connected")
                        if self.port != 0:
                            self.port.close()
                            self.port = 0
                    pass  
                break
                
        vid.release()
        cv2.destroyAllWindows()
        self.imgVideo.clear()
        if self.keyPupilsProcess == True:
            self.processRecord()

    def setPhoto(self):
        widthVideo = self.imgVideo.width()
        heightVideo = self.imgVideo.height()
        image = imutils.resize(self.frameData, width=widthVideo, height=heightVideo)
        image = QtGui.QImage(image, image.shape[1],image.shape[0],image.strides[0], QtGui.QImage.Format_BGR888) 
        self.imgVideo.setPixmap(QtGui.QPixmap.fromImage(image))

## Pupils Process
    # Recording Pupils
    def processRecord(self):
        self.Worker1 = ThreadCameraGet(self.set_camera.currentIndex())
        self.Worker1.start()
        if not self.Worker1.stopped:
            self.Worker2 = ThreadShowFrame(self.Worker1.frame, self.imgVideo.width(), self.imgVideo.height())
            self.Worker2.start()
        self.out = cv2.VideoWriter(self.filenameVideo, cv2.VideoWriter_fourcc(*'XVID'), self.fps_in_recording, (640, 480))
        self.status_bar.setText("Recording Process")
        countFrame = 0
        keyProcess = 0
        elapsed_time = 0
        self.a_x = 0
        self.c_x = 0
        expectedTime = 1/self.fps_in_recording
        realTime = datetime.now()
        while(True):
            elapsed_time = (datetime.now() - realTime).total_seconds()
            if elapsed_time >= expectedTime and keyProcess == 0:
                try:
                    self.port.write(str.encode('1'))
                    keyProcess = 1
                    print("IR LED ON " + str(elapsed_time))
                except:
                    keyProcess = 5
                    self.device_status_control.setText("Not Connected")
                    self.device_status_connection.setText("Not Connected")
                    if self.port != 0:
                        self.port.close()
                        self.port = 0
                    pass
            elapsed_time = (datetime.now() - realTime).total_seconds()
            if elapsed_time >= self.whiteLedON and keyProcess == 1:
                try:
                    self.port.write(str.encode('2'))
                    self.a_x = float('%.3f'%(elapsed_time))
                    keyProcess = 2
                    print("White LED ON " + str(elapsed_time))
                except:
                    keyProcess = 5
                    self.device_status_control.setText("Not Connected")
                    self.device_status_connection.setText("Not Connected")
                    if self.port != 0:
                        self.port.close()
                        self.port = 0
                    pass
            elapsed_time = (datetime.now() - realTime).total_seconds()
            if elapsed_time >= self.whiteLedOFF and keyProcess == 2:
                try:
                    self.port.write(str.encode('3'))
                    self.c_x = float('%.3f'%(elapsed_time))
                    keyProcess = 3
                    print("White LED OFF " + str(elapsed_time))
                except:
                    keyProcess = 5
                    self.device_status_control.setText("Not Connected")
                    self.device_status_connection.setText("Not Connected")
                    if self.port != 0:
                        self.port.close()
                        self.port = 0
                    pass
            elapsed_time = (datetime.now() - realTime).total_seconds()
            if elapsed_time >= self.duration and keyProcess == 3:
                try:
                    self.port.write(str.encode('4'))
                    keyProcess = 4
                    print("IR LED OFF " + str(elapsed_time))
                except:
                    keyProcess = 5
                    self.device_status_control.setText("Not Connected")
                    self.device_status_connection.setText("Not Connected")
                    if self.port != 0:
                        self.port.close()
                        self.port = 0
                    pass

            self.frame = self.Worker1.frame
            self.image = cv2.flip(self.frame, +1)
            self.Worker2.widthVideo = self.imgVideo.width()
            self.Worker2.heightVideo = self.imgVideo.height()
            self.Worker2.frame = self.image
            elapsed_time = (datetime.now() - realTime).total_seconds()
            if elapsed_time >= expectedTime:
                countFrame += 1
                expectedTime += (1/self.fps_in_recording)
                self.out.write(self.image)
                if countFrame % self.actualFPSDisplay == 0:
                    self.imgVideo.setPixmap(QtGui.QPixmap.fromImage(self.Worker2.image))
                    self.status_timer.setText(str('%.3f'%(elapsed_time)))
            key = cv2.waitKey(1) & 0xFF
            if elapsed_time >= self.duration:
                try:
                    self.port.write(str.encode('6'))
                except:
                    pass
                finally:
                    self.Worker1.stop()
                    self.Worker2.stop()
                    self.Worker1.quit()
                    self.Worker2.quit()
                    self.status_timer.setText("0")
                    self.total_frame.setText(str(countFrame))
                break

            if self.Worker1.stopped:
                try:
                    self.port.write(str.encode('6'))
                except:
                    pass
                finally:
                    self.Worker1.stop()
                    self.Worker2.stop()
                    self.Worker1.quit()
                    self.Worker2.quit()
                    self.openSetting()
                    self.total_frame.setText("0")
                    self.keyPupilsProcess = False
                    self.set_camera.setEnabled(True)
                    self.status_bar.setText("Not Processing")
                    self.device_status_camera.setText("Not Connected")
                    self.btn_on_off_camera.setText("Start Cam")
                    self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                    self.status_timer.setText("0")
                break

            if self.keyPupilsProcess == False:
                try:
                    self.port.write(str.encode('6'))
                except:
                    pass
                finally:
                    self.Worker1.stop()
                    self.Worker2.stop()
                    self.Worker1.quit()
                    self.Worker2.quit()
                    self.openSetting()
                    self.total_frame.setText("0")
                    self.status_timer.setText("0")
                    self.status_bar.setText("Not Processing")
                    self.btn_on_off_camera.setText("Start Cam")
                    self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                    self.set_camera.setEnabled(True)
                    break
        
        cv2.destroyAllWindows()
        self.out.release()
        self.imgVideo.clear()

        if self.keyPupilsProcess == True:
            self.processMeasurement()

    # Pengukuran Pupils
    def processMeasurement(self):
        # Default 
        fr = 0
        sec = 0
        count = 0
        count_frame = 0
        avg_ref = 0
        ref_lst = []
        self.diameter = []
        self.timePupils = []
        self.status_bar.setText("Measurement Process")

        vid = cv2.VideoCapture(self.filenameVideo)
        totalFrame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.propFPS = vid.get(cv2.CAP_PROP_FPS)
        timeVideo = round(totalFrame/self.propFPS)
        socondPerFrame = timeVideo/totalFrame

        while(vid.isOpened()):
            sec += socondPerFrame
            img, self.image = vid.read()

            sticker = self.image[240:640, 0:640]

            if fr < 30:
                gray_s = cv2.cvtColor(sticker, cv2.COLOR_BGR2GRAY)
                gaussian_s = cv2.GaussianBlur(gray_s, (17, 17), 0)
                circles2 = cv2.HoughCircles(gaussian_s,
                                            cv2.HOUGH_GRADIENT,
                                            1.5, minDist=100,
                                            param1=30, param2=65,
                                            minRadius=50, maxRadius=100)

                # Deteksi Lingkaran
                if circles2 is not None:
                    circles2 = np.round(circles2[0, :]).astype("int")
                    for (x2, y2, r2) in circles2:
                        cv2.circle(sticker, (x2, y2), r2, (0, 255, 0), 2)
                        asli = 6 
                        reference = asli / r2  
                        ref_lst.append(reference)
                        sum_ref = sum(ref_lst)
                        avg_ref = sum_ref / len(ref_lst)
                
                fr += 1

            else:
                # Eye detection
                eye = self.image[0:240, 0:640]

                gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
                threshold = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 181, 35)
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.Canny(threshold, 50, 100)
                dilated = cv2.dilate(edges, None, iterations=2)

                cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                center = None

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(eye, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    if 10 < radius < 100:
                        cv2.circle(eye, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                    if avg_ref == 0:
                        self.btn_on_off_camera.setText("Start Cam")
                        self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                        self.status_bar.setText("Not Processing")
                        self.status_timer.setText("0")
                        self.set_camera.setEnabled(True)
                        self.keyPupilsProcess = False
                        self.openHideButtonAnalytics()
                        self.btn_save_data.setEnabled(True)
                        self.btn_save_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                        self.openSetting()
                        break

                    radius_mm = radius * avg_ref

                    # Menyimpan Pengukuran & Waktu
                    self.diameter.insert(count, radius_mm)
                    self.timePupils.insert(count, float('%.3f'%(sec)))
                    count = count + 1
            
            count_frame += 1
            self.frameData = np.hstack([self.image])
            self.setPhoto()
            
            self.status_timer.setText(str('%.3f'%(sec)))
         
            key = cv2.waitKey(0) & 0xFF
            if count_frame == totalFrame:
                self.btn_on_off_camera.setText("Start Cam")
                self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.status_bar.setText("Not Processing")
                self.status_timer.setText("0")
                self.set_camera.setEnabled(True)
                self.keyPupilsProcess = False
                if self.diameter == [] and self.timePupils == []:
                    break
                self.getData()
                self.openHideButtonAnalytics()
                self.btn_save_data.setEnabled(True)
                self.btn_save_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.openSetting()
                break

            if self.keyPupilsProcess == False:
                self.btn_on_off_camera.setText("Start Cam")
                self.btn_on_off_camera.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.status_bar.setText("Not Processing")
                self.status_timer.setText("0")
                self.set_camera.setEnabled(True)
                self.openHideButtonAnalytics()
                self.btn_save_data.setEnabled(True)
                self.btn_save_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.openSetting()
                break
        
        vid.release()
        cv2.destroyAllWindows()
        self.imgVideo.clear()

## Video Process
    def getvideoProcess(self):
        # Penandaan Waktu Kapan Nyala LED Putih Sulit Dilakukan Pada Video
        self.filename = QtWidgets.QFileDialog.getOpenFileName(filter = "(*.avi)")[0]
        filepathname = os.path.basename(self.filename)
        self.filebasename = os.path.splitext(filepathname)[0]
        fileXML = "MeasurementData/" + str(self.filebasename) + '.xml'
        if os.path.exists(fileXML) == True:
            treeXMLData = ET.parse(fileXML)
            LedON = treeXMLData.find('LedON').text
            LedOFF = treeXMLData.find('LedOFF').text
            self.a_x = float(LedON)
            self.c_x = float(LedOFF)
            self.outDataXML(fileXML)
        elif os.path.exists(fileXML) == False:
            self.a_x = 2
            self.c_x = 8
        
        if self.filename == '':
            self.keyVideoProcess = False
            self.status_bar.setText("Not Processing")
        else:
            self.set_camera.setEnabled(False)
            self.hideSetting()
            self.videoProcess()
    
    def videoProcess(self):
        vid = cv2.VideoCapture(self.filename)
        totalFrame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.propFPS = vid.get(cv2.CAP_PROP_FPS)
        timeVideo = round(totalFrame/self.propFPS)
        socondPerFrame = timeVideo/totalFrame
        fr = 0
        sec = 0
        count = 0
        countFrame = 0
        avg_ref = 0
        ref_lst = []
        self.diameter = []
        self.timePupils = []

        while(vid.isOpened()):
            sec += socondPerFrame
            img, self.image = vid.read()

            sticker = self.image[240:640, 0:600]

            if fr < 30:
                gray_s = cv2.cvtColor(sticker, cv2.COLOR_BGR2GRAY)
                gaussian_s = cv2.GaussianBlur(gray_s, (17, 17), 0)
                circles2 = cv2.HoughCircles(gaussian_s,
                                            cv2.HOUGH_GRADIENT,
                                            1.5, minDist=100,
                                            param1=30, param2=65,
                                            minRadius=50, maxRadius=100)

                # Deteksi Lingkaran
                if circles2 is not None:
                    circles2 = np.round(circles2[0, :]).astype("int")
                    for (x2, y2, r2) in circles2:
                        cv2.circle(sticker, (x2, y2), r2, (0, 255, 0), 2)
                        asli = 6 
                        reference = asli / r2 
                        ref_lst.append(reference)
                        sum_ref = sum(ref_lst)
                        avg_ref = sum_ref / len(ref_lst)
                
                fr += 1

            else:
                # Eye detection
                eye = self.image[0:240, 0:640]
                gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
                threshold = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 181, 35)
                # _, threshold = cv2.threshold(gaussian, 10, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.Canny(threshold, 50, 100)
                dilated = cv2.dilate(edges, None, iterations=2)

                cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                center = None

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(eye, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    if 10 < radius < 100:
                        cv2.circle(eye, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                    if avg_ref == 0:
                        self.status_bar.setText("Not Processing")
                        self.status_timer.setText("0")
                        self.set_camera.setEnabled(True)
                        self.keyVideoProcess = False
                        self.openHideButtonAnalytics()
                        self.btn_update_data.setEnabled(True)
                        self.btn_update_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                        self.openSetting()
                        break

                    radius_mm = radius * avg_ref

                    # Menyimpan Pengukuran & Waktu
                    self.diameter.insert(count, radius_mm)
                    self.timePupils.insert(count, float('%.3f'%(sec)))
                    count = count + 1

            countFrame += 1
            self.frameData = np.hstack([self.image])
            self.setPhoto()
            
            self.status_timer.setText(str('%.3f'%(sec)))

            key = cv2.waitKey(0) & 0xFF
            if countFrame == totalFrame:
                self.status_bar.setText("Not Processing")
                self.status_timer.setText("0")
                self.set_camera.setEnabled(True)
                self.keyVideoProcess = False
                if self.diameter == [] and self.timePupils == []:
                    break
                self.getData()
                self.openHideButtonAnalytics()
                self.btn_update_data.setEnabled(True)
                self.btn_update_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.openSetting()
                break

            if self.keyVideoProcess == False:
                self.status_bar.setText("Not Processing")
                self.status_timer.setText("0")
                self.set_camera.setEnabled(True)
                self.openHideButtonAnalytics()
                self.btn_update_data.setEnabled(True)
                self.btn_update_data.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
                self.openSetting()
                break

        vid.release()
        cv2.destroyAllWindows()
        self.imgVideo.clear()

## Display Grafik
    def getData(self):
        self.lstkec = []
        self.lsttm = []
        self.lst = []

        def hitung(dataset, interval=1):
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return diff

        df = pd.DataFrame(self.diameter)  # dataframe (150,1)

        # extracting time column
        time_df = pd.DataFrame(self.timePupils) # dataframe (150, 1)
        time_arr = time_df.to_numpy()           # ndarray (150, 1)

        # extracting diameter column
        diameter_df = df.iloc[:, 0]             # series (150, )
        diameter_avg = diameter_df.rolling(window=self.avgDiameter, min_periods=1).mean() # series (150,)
        diameter_arr = diameter_avg.to_numpy()  # ndarray (150, ) windowing diameter
        
        # counting difference between each row (diameter)
        diff_diameter = hitung(diameter_arr)            # list: 149
        pd_diff_diameter = pd.DataFrame(diff_diameter)  # dataframe (149,1)

        # counting difference between each row (time)
        diff_time = hitung(time_arr)            # list:  149
        pd_diff_time = pd.DataFrame(diff_time)  # dataframe (149,1)

        # KECEPATAN
        self.lstkec = pd_diff_diameter / pd_diff_time
        pd_lstkec = pd.DataFrame(self.lstkec)   # dataframe (149,1)
        avg_kec = pd_lstkec.rolling(window=self.avgVelocity, min_periods=1).mean() # dataframe (149,1)
        avg_kec_hasil = avg_kec.to_numpy()  # ndarray (149,1)

        # AKSELERASI
        diff_kec = hitung(avg_kec_hasil)
        pd_diff_kec = pd.DataFrame(diff_kec)
        self.lstaks = pd_diff_kec / pd_diff_time
        pd_lstaks = pd.DataFrame(self.lstaks).fillna(0)
        avg_aks = pd_lstaks.rolling(window=self.avgAcceleration, min_periods=1).mean()
        avg_aks_hasil = avg_aks.to_numpy()

        times_d = np.array([self.timePupils]).T              # ndarray (150, 1)
        times_d = times_d[:-1].copy()
        diameter_p = np.array([self.diameter]).T             # ndarray (150,1)
        diameter_p = diameter_p[:-1].copy()
        np_lstkec = pd_lstkec.to_numpy()                # ndarray (149,1)
        np_lstaks = pd_lstaks.to_numpy()                # ndarray (149,1)
        diameter_arr_t = np.array([diameter_arr]).T     # ndarray (150,1)
        diameter_arr_t = diameter_arr_t[:-1].copy()

        a = np.concatenate((times_d,        # ndarray (149, 1)
                            diameter_p,     # ndarray (149,1)
                            np_lstkec,      # ndarray (149,1)
                            np_lstaks,      # ndarray (149,1)
                            diameter_arr_t, # ndarray (149,1)
                            avg_kec_hasil,  # ndarray (149,1)
                            avg_aks_hasil), # ndarray (149,1)
                            axis=1)

        np.savetxt("TemporaryData/temporaryData.csv", a, delimiter=',', header="Time,Diameter,Kecepatan,Akselerasi,AvgD,AvgK,AvgA",
                   comments="", fmt='%10.3f')

        self.df = pd.read_csv("TemporaryData/temporaryData.csv", encoding = 'utf-8').fillna(0)
        self.getSelectGrap = 1
        self.Update('bmh')

    def Update(self, value):
        plt.clf()
        plt.style.use(value)
        try:
            self.canv.axes.cla()
            self.canv1.axes.cla()
            self.canv2.axes.cla()

            self.canv.draw()
            self.canv1.draw()
            self.canv2.draw()

            self.grap_pengukuran.removeWidget(self.canv)
            self.grap_kecepatan.removeWidget(self.canv1)
            self.grap_akselerasi.removeWidget(self.canv2)

            self.canv = None
            self.canv1 = None
            self.canv2 = None
        except Exception as e:
            pass

        self.canv = MatplotlibCanvas(self)
        self.canv1 = MatplotlibCanvas(self)
        self.canv2 = MatplotlibCanvas(self)

        self.grap_pengukuran.addWidget(self.canv)
        self.grap_kecepatan.addWidget(self.canv1)
        self.grap_akselerasi.addWidget(self.canv2)

        self.canv.axes.cla()
        ax = self.canv.axes

        x_filt = self.df.Time[self.df.Time > 0] 
        self.x_filt = x_filt
        y_d_filt = self.df.AvgD[self.df.Time > 0]
        y_k_filt = self.df.AvgK[self.df.Time > 0]
        self.y_k_filt = y_k_filt
        y_a_filt = self.df.AvgA[self.df.Time > 0]
        self.y_a_filt = y_a_filt

        y_d_filt_sblm_led = y_d_filt[0:self.avgDiameter]
        y_k_filt_sblm_led = y_k_filt[self.df.Time < self.a_x]
        y_a_filt_sblm_led = y_a_filt[self.df.Time < self.a_x]

        avg_ydfilt_sblm_led = sum(y_d_filt_sblm_led) / len(y_d_filt_sblm_led)
        avg_ydfilt_sblm_led = round(avg_ydfilt_sblm_led, 3)
        self.df.loc[0:self.avgDiameter, 'AvgD'] = avg_ydfilt_sblm_led
        y_d_filt = self.df.AvgD[self.df.Time > 0]

        self.hitungIndex = (self.a_x - 1) * self.propFPS

        avg_ykfilt_sblm_led = sum(y_k_filt_sblm_led) / len(y_k_filt_sblm_led)
        avg_ykfilt_sblm_led = round(avg_ykfilt_sblm_led, 3)
        self.df.loc[0:self.hitungIndex, 'AvgK'] = avg_ykfilt_sblm_led
        y_k_filt = self.df.AvgK[self.df.Time > 0]

        avg_yafilt_sblm_led = sum(y_a_filt_sblm_led) / len(y_a_filt_sblm_led)
        avg_yafilt_sblm_led = round(avg_yafilt_sblm_led, 3)
        self.df.loc[0:self.hitungIndex, 'AvgA'] = avg_yafilt_sblm_led
        y_a_filt = self.df.AvgA[self.df.Time > 0]

        d_awal = y_d_filt[self.df.Time >= self.a_x].index[0]
        d_awal_hasil = y_d_filt[d_awal]
        d_awal = round(d_awal,3)
        self.hasil_d.setText(str(d_awal_hasil))

        ax.plot(x_filt, y_d_filt, c='k', lw='0.7')
        # Point (a) grafik pengukuran
        # Point (a) ketika LED putih pertama nyala
        a_y = np.interp(self.a_x , x_filt, y_d_filt)

        # Point (c) graph pengukuran
        # Point (c) ketika LED putih mati
        c_y = np.interp(self.c_x, x_filt, y_d_filt)

        min_y = y_d_filt.min()
        j = np.where(y_d_filt == min_y)
        j_new = j[0][0] # index
        min_x = x_filt.iloc[j_new]

        ax.plot(0, a_y)
        ax.plot(self.a_x, a_y, 'go')
        ax.plot(self.c_x, c_y, 'ro')
        ax.plot(min_x, min_y, 'bo')
        ax.set_xlabel('Time after stimulus (sec)')
        ax.set_ylabel('Pupil diameter (mm)')
        ax.axvline(self.a_x, 0, label='LED Putih ON', c='g', ls='--', lw='0.7')
        ax.axvline(self.c_x, 0, label='LED Putih OFF', c='r', ls='--', lw='0.7')

        self.canv1.axes.cla()
        ax1 = self.canv1.axes

        ax1.plot(x_filt, y_k_filt, c='b', lw='0.7')

        # Point (a) grafik kecepatan
        # Point (a) ketika LED putih pertama nyala
        a_y = np.interp(self.a_x, x_filt, y_k_filt)

        # Point (c) graph kecepatan
        # Point (c) ketika LED putih mati
        c_y = np.interp(self.c_x, x_filt, y_k_filt)

        min_y = y_k_filt.min()
        self.mcvmin_y = min_y
        j = np.where(y_k_filt == min_y)
        j_new = j[0][0]
        min_x = x_filt.iloc[j_new]
        self.min_kx = min_x

        y_k_nyala = y_k_filt[self.df.Time >= self.a_x].index[0]
        x_k_nyala = x_filt[self.df.Time >= self.a_x].index[0]

        y_k_filt_min = y_k_filt.min()
        y_k_filt_min_idx = np.where(y_k_filt == y_k_filt_min)
        y_k_filt_min_idx = y_k_filt_min_idx[0][0]
        x_k_filt_min = x_filt.iloc[y_k_filt_min_idx]
        min_y_kec = y_k_filt[self.df.Time == x_k_filt_min].index[0]

        data_turun_ky = y_k_filt[y_k_nyala : min_y_kec]

        min_x_kec = x_filt[self.df.Time == x_k_filt_min].index[0]
        data_turun_kx = x_filt[x_k_nyala : min_x_kec]

        for idx_turun, val_turun in enumerate(data_turun_ky):
            if val_turun < 0:
                turuncuram_y = val_turun
                idx_y_k = idx_turun
                turuncuram_x = data_turun_kx.iloc[idx_y_k]
                break

        mcv = turuncuram_y - y_k_filt_min
        self.hasil_mcv.setText(str('%.3f'%(mcv)))

        tc = self.min_kx - turuncuram_x
        self.hasil_tc.setText(str('%.3f'%(tc)))

        ax1.plot(0, 0)
        ax1.plot(self.a_x, a_y, 'go')
        ax1.plot(self.c_x, c_y, 'ro')
        ax1.plot(min_x, min_y, 'bo')
        ax1.set_xlabel('Time after stimulus (sec)')
        ax1.set_ylabel('Velocity (mm/sec)')
        ax1.axvline(self.a_x, 0, label='LED Putih ON', c='g', ls='--', lw='0.7')
        ax1.axvline(self.c_x, 0, label='LED Putih OFF', c='r', ls='--', lw='0.7')
        self.otomaticGrapMCV = ax1.scatter(turuncuram_x, turuncuram_y, c="black")

        self.canv2.axes.cla()
        ax2 = self.canv2.axes

        ax2.plot(x_filt, y_a_filt, c='r', lw='0.7')

        # Point (a) grafik akselerasi
        # Point (a) ketika LED putih pertama nyala
        a_y = np.interp(self.a_x, x_filt, y_a_filt)

        # Point (c) graph akselerasi
        # Point (c) ketika LED putih mati
        c_y = np.interp(self.c_x, x_filt, y_a_filt)

        min_y = y_a_filt.min()
        self.mcamin_y = min_y
        j = np.where(y_a_filt == min_y)
        j_new = j[0][0]
        min_x = x_filt.iloc[j_new]

        y_a_nyala = y_k_filt[self.df.Time >= self.a_x].index[0]
        x_a_nyala = x_filt[self.df.Time >= self.a_x].index[0]

        y_a_filt_min = y_a_filt.min()
        y_a_filt_min_idx = np.where(y_a_filt == y_a_filt_min)
        y_a_filt_min_idx = y_a_filt_min_idx[0][0]
        x_a_filt_min = x_filt.iloc[y_a_filt_min_idx]
        min_y_aks = y_a_filt[self.df.Time == x_a_filt_min].index[0]

        data_turun_ay = y_a_filt[y_a_nyala : min_y_aks]

        min_x_aks = x_filt[self.df.Time == x_a_filt_min].index[0]
        data_turun_ax = x_filt[x_a_nyala : min_x_aks]

        for idx_turun_a, val_turun_a in enumerate(data_turun_ay):
            if val_turun_a < 0:
                turuncuram_ay = val_turun_a
                idx_y_turun_a = idx_turun_a
                turuncuram_ax = data_turun_ax.iloc[idx_y_turun_a]
                break

        mca = turuncuram_ay - y_a_filt_min
        self.hasil_mca.setText(str('%.3f'%(mca)))

        ax2.plot(0, 0)
        ax2.plot(self.a_x, a_y, 'go')
        ax2.plot(self.c_x, c_y, 'ro')
        ax2.plot(min_x, min_y, 'bo')
        ax2.set_xlabel('Time after stimulus(sec)')
        ax2.set_ylabel('Acceleration (mm/sec²)')
        ax2.axvline(self.a_x, 0, label='LED Putih ON', c='g', ls='--', lw='0.7')
        ax2.axvline(self.c_x, 0, label='LED Putih OFF', c='r', ls='--', lw='0.7')
        self.otomaticGrapMCA = ax2.scatter(turuncuram_ax, turuncuram_ay, c='black')

        y_kec = self.df.Kecepatan
        y_aks = self.df.Akselerasi

        acv_nyala = y_kec[self.df.Time >= self.a_x].index[0]
        acv_mati = y_kec[self.df.Time >= self.c_x].index[0]
        data_acv = y_kec[acv_nyala : acv_mati]
        acv = (sum(abs(data_acv))/len(data_acv)) 
        
        self.hasil_acv.setText(str('%.3f'%(acv)))

        aca_nyala = y_aks[self.df.Time >= self.a_x].index[0]
        aca_mati = y_aks[self.df.Time >= self.c_x].index[0]
        data_aca = y_aks[aca_nyala : aca_mati]
        aca = (sum(abs(data_aca))/len(data_aca))
        
        self.hasil_aca.setText(str('%.3f'%(aca)))

        # latency
        latency = turuncuram_x -self.a_x
        self.hasil_tl.setText(str('%.3f'%(latency)))

        self.canv.draw()
        self.canv1.draw()
        self.canv2.draw()

## Pengisian Data
    def saveData(self):
        Nama = self.int_nama.text()
        Nik_Id= self.int_nik.text()
        TempatLahir = self.int_tempat_lahir.text()
        TanggalLahir = self.int_tanggal_lahir.text()
        JenisKelamin = self.int_jenis_kelamin.text()
        Pekerjaan = self.int_pekerjaan.text()
        Pendidikan = self.int_pendidikan.text()
        Agama = self.int_agama.text()
        Comment = self.textEdit.toPlainText()
        diameterBaseLine = self.hasil_d.text()
        timeLatency = self.hasil_tl.text()
        timeContraction = self.hasil_tc.text()
        maxContractionVelocity = self.hasil_mcv.text()
        avgContractionVelocity = self.hasil_acv.text()
        maxContractionAcceleration = self.hasil_mca.text()
        avgContractionAcceleration = self.hasil_aca.text()
        LedON = str(self.a_x)
        LedOFF = str(self.c_x)

        filename = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.xml'

        data = ET.Element('DataPasien')
        
        s_elem1 = ET.SubElement(data, 'Nama')
        s_elem2 = ET.SubElement(data, 'NIK')
        s_elem3 = ET.SubElement(data, 'Tempat_Lahir')
        s_elem4 = ET.SubElement(data, 'Tanggal_Lahir')
        s_elem5 = ET.SubElement(data, 'Jenis_Kelamin')
        s_elem6 = ET.SubElement(data, 'Pekerjaan')
        s_elem7 = ET.SubElement(data, 'Pendidikan')
        s_elem8 = ET.SubElement(data, 'Agama')
        s_elem9 = ET.SubElement(data, 'Comment')
        s_elem10 = ET.SubElement(data, 'LedON')
        s_elem11 = ET.SubElement(data, 'LedOFF')
        s_elem12 = ET.SubElement(data, 'D')
        s_elem13 = ET.SubElement(data, 'TL')
        s_elem14 = ET.SubElement(data, 'TC')
        s_elem15 = ET.SubElement(data, 'MCV')
        s_elem16 = ET.SubElement(data, 'ACV')
        s_elem17 = ET.SubElement(data, 'MCA')
        s_elem18 = ET.SubElement(data, 'ACA')
        
        s_elem1.text = Nama
        s_elem2.text = Nik_Id
        s_elem3.text = TempatLahir
        s_elem4.text = TanggalLahir
        s_elem5.text = JenisKelamin
        s_elem6.text = Pekerjaan
        s_elem7.text = Pendidikan
        s_elem8.text = Agama
        s_elem9.text = Comment
        s_elem10.text = LedON
        s_elem11.text = LedOFF
        s_elem12.text = diameterBaseLine
        s_elem13.text = timeLatency
        s_elem14.text = timeContraction
        s_elem15.text = maxContractionVelocity
        s_elem16.text = avgContractionVelocity
        s_elem17.text = maxContractionAcceleration
        s_elem18.text = avgContractionAcceleration

        b_xml = ET.tostring(data)
        
        with open(filename, "wb") as f:
            f.write(b_xml)

        self.clearData()

        src = "TemporaryData/temporaryData.csv"
        filenamecsv = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.csv'
        filenameavi = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.avi'

        shutil.copy(src,filenamecsv)
        shutil.copy(self.filenameVideo, filenameavi)

        self.hideButtonAnalytics()

## Update Data
    def updateData(self):
        fileAVI = "MeasurementData/" + str(self.filebasename) + '.avi'
        fileCSV = "MeasurementData/" + str(self.filebasename) + '.csv'
        fileXML = "MeasurementData/" + str(self.filebasename) + '.xml'

        Nama = self.int_nama.text()
        Nik_Id= self.int_nik.text()
        TempatLahir = self.int_tempat_lahir.text()
        TanggalLahir = self.int_tanggal_lahir.text()

        filename = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.xml'

        data = ET.Element('DataPasien')
        
        s_elem1 = ET.SubElement(data, 'Nama')
        s_elem2 = ET.SubElement(data, 'NIK')
        s_elem3 = ET.SubElement(data, 'Tempat_Lahir')
        s_elem4 = ET.SubElement(data, 'Tanggal_Lahir')
        s_elem5 = ET.SubElement(data, 'Jenis_Kelamin')
        s_elem6 = ET.SubElement(data, 'Pekerjaan')
        s_elem7 = ET.SubElement(data, 'Pendidikan')
        s_elem8 = ET.SubElement(data, 'Agama')
        s_elem9 = ET.SubElement(data, 'Comment')
        s_elem10 = ET.SubElement(data, 'LedON')
        s_elem11 = ET.SubElement(data, 'LedOFF')
        s_elem12 = ET.SubElement(data, 'D')
        s_elem13 = ET.SubElement(data, 'TL')
        s_elem14 = ET.SubElement(data, 'TC')
        s_elem15 = ET.SubElement(data, 'MCV')
        s_elem16 = ET.SubElement(data, 'ACV')
        s_elem17 = ET.SubElement(data, 'MCA')
        s_elem18 = ET.SubElement(data, 'ACA')
        
        s_elem1.text = Nama
        s_elem2.text = Nik_Id
        s_elem3.text = TempatLahir
        s_elem4.text = TanggalLahir
        s_elem5.text = self.int_jenis_kelamin.text()
        s_elem6.text = self.int_pekerjaan.text()
        s_elem7.text = self.int_pendidikan.text()
        s_elem8.text = self.int_agama.text()
        s_elem9.text = self.textEdit.toPlainText()
        s_elem10.text = str(self.a_x)
        s_elem11.text = str(self.c_x)
        s_elem12.text = self.hasil_d.text()
        s_elem13.text = self.hasil_tl.text()
        s_elem14.text = self.hasil_tc.text()
        s_elem15.text = self.hasil_mcv.text()
        s_elem16.text = self.hasil_acv.text()
        s_elem17.text = self.hasil_mca.text()
        s_elem18.text = self.hasil_aca.text()

        b_xml = ET.tostring(data)
        
        with open(filename, "wb") as f:
            f.write(b_xml)

        self.clearData()

        if os.path.exists(fileCSV) == True:
            os.remove(fileCSV)

        src = "TemporaryData/temporaryData.csv"
        filenamecsv = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.csv'
        filenameavi = "MeasurementData/" + Nik_Id + "_" + Nama + "_" + TempatLahir + "_" + TanggalLahir + '.avi'

        shutil.copy(src, filenamecsv)

        if filename != fileXML:
            shutil.copy(fileAVI, filenameavi)

        if filename != fileXML:
            if os.path.exists(fileAVI) == True:
                os.remove(fileAVI)
            if os.path.exists(fileXML) == True:
                os.remove(fileXML)

        self.hideButtonAnalytics()

## Pembukaan Data
    def outDataXML(self, path):
        fileXML = path
        treeXMLData = ET.parse(fileXML)

        self.int_nama.setText(treeXMLData.find('Nama').text)
        self.int_nik.setText(treeXMLData.find('NIK').text)
        self.int_tempat_lahir.setText(treeXMLData.find('Tempat_Lahir').text)
        self.int_tanggal_lahir.setText(treeXMLData.find('Tanggal_Lahir').text)
        self.int_jenis_kelamin.setText(treeXMLData.find('Jenis_Kelamin').text)
        self.int_pekerjaan.setText(treeXMLData.find('Pekerjaan').text)
        self.int_pendidikan.setText(treeXMLData.find('Pendidikan').text)
        self.int_agama.setText(treeXMLData.find('Agama').text)
        self.textEdit.setText(treeXMLData.find('Comment').text)
        self.hasil_d.setText(treeXMLData.find('D').text)
        self.hasil_tl.setText(treeXMLData.find('TL').text)
        self.hasil_tc.setText(treeXMLData.find('TC').text)
        self.hasil_mcv.setText(treeXMLData.find('MCV').text)
        self.hasil_acv.setText(treeXMLData.find('ACV').text)
        self.hasil_mca.setText(treeXMLData.find('MCA').text)
        self.hasil_aca.setText(treeXMLData.find('ACA').text)

## Menghapus Data
    def clearData(self):       
        self.canv.axes.cla()
        self.canv1.axes.cla()
        self.canv2.axes.cla()

        self.canv.draw()
        self.canv1.draw()
        self.canv2.draw()

        self.getSelectGrap = 0

        self.hasil_d.setText("0")
        self.hasil_tc.setText("0")
        self.hasil_tl.setText("0")
        self.hasil_mcv.setText("0")
        self.hasil_mca.setText("0")
        self.hasil_aca.setText("0")
        self.hasil_acv.setText("0")
        self.int_nama.setText("")
        self.int_nik.setText("")
        self.int_tempat_lahir.setText("")
        self.int_tanggal_lahir.setText("")
        self.int_jenis_kelamin.setText("")
        self.int_pekerjaan.setText("")
        self.int_pendidikan.setText("")
        self.int_agama.setText("")
        self.textEdit.setText("")

        self.hideButtonAnalytics()

## Device Connecting
    def deviceConnecting(self):
        self.btn_connecting.setEnabled(False)
        self.btn_connecting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
        if self.port != 0:
            self.port.close()
            self.port = 0
        self.connectionTask = TaskThread(varCOM=self.set_port_device.currentText())
        self.connectionTask.notifyProgressBar.connect(self.updateProgressBar)
        self.connectionTask.notifyConnect.connect(self.updateStatusConnection)
        self.connectionTask.valueConnect.connect(self.updateCOM)
        self.connectionTask.start()

    def updateProgressBar(self, valueBar):
        self.progress_connection.setValue(valueBar)
        if self.progress_connection.value() == 100:
            self.progress_connection.setValue(0)
            self.btn_connecting.setEnabled(True)
            self.btn_connecting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;\n}QPushButton::hover{background-color: rgb(57, 57, 63);}")
    
    def updateStatusConnection(self, connection):
        self.device_status_connection.setText(connection)
        self.device_status_control.setText(connection)

    def updateCOM(self, datacom):
        self.port = serial.Serial(datacom, 9600)

## Process MCV MCA
    ### MCV PROCESS ###
    def processMCV(self):
        global scatterMCV
        if self.select_MCV.text() == "Select MCV" and self.select_MCA.text() == "Select MCA" and self.getSelectGrap == 1:
            self.select_MCV.setText("Deselect MCV")
            self.select_MCV.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
            if self.otomaticGrapMCV != 0:
                self.otomaticGrapMCV.remove()
                self.otomaticGrapMCV = 0
                self.canv1.draw()
            if scatterMCV != 0:
                scatterMCV.remove()
                scatterMCV = 0
                self.canv1.draw()

            def mouse_move(event):
                if not event.inaxes: # klo di luar axes
                    return
                x, y = event.xdata, event.ydata
                indx = min(np.searchsorted(self.x_filt, x), len(self.x_filt) - 1) # data x diurutin, supaya cursor ngikut sumbu x
                if indx < self.hitungIndex:
                    x = self.x_filt[self.hitungIndex]
                    y = self.y_k_filt[self.hitungIndex]
                else:
                    x = self.x_filt[indx]
                    y = self.y_k_filt[indx]
                axh = self.canv1.axes.axhline(color='k')
                axv = self.canv1.axes.axvline(color='k')
                axh.set_ydata(y)
                axv.set_xdata(x)
                self.canv1.draw()
                axh.remove()
                axv.remove()

            def onclick(event):
                global scatterMCV
                if scatterMCV != 0:
                    scatterMCV.remove()
                x, y = event.xdata, event.ydata
                indx = min(np.searchsorted(self.x_filt, x), len(self.x_filt) - 1) # data x diurutin, supaya cursor ngikut sumbu x
                if indx < self.hitungIndex:
                    x = self.x_filt[self.hitungIndex]
                    y = self.y_k_filt[self.hitungIndex]
                else:
                    x = self.x_filt[indx]
                    y = self.y_k_filt[indx]
                latency = x - self.a_x
                mcv = y - self.mcvmin_y
                tc = self.min_kx - x
                self.hasil_tc.setText(str('%.3f'%(tc)))
                self.hasil_tl.setText(str('%.3f'%(latency)))
                self.hasil_mcv.setText(str('%.3f'%(mcv)))
                scatterMCV = self.canv1.axes.scatter(x, y, c='black')
                self.canv1.draw()
            
            self.onClicked = self.canv1.fig.canvas.mpl_connect('button_press_event', onclick)
            self.mouseMove = self.canv1.fig.canvas.mpl_connect('motion_notify_event', mouse_move)

        elif self.select_MCV.text() == "Deselect MCV":
            self.select_MCV.setText("Select MCV")
            self.select_MCV.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;} QPushButton::hover{background-color: rgb(57, 57, 63);}")
            axh = self.canv1.axes.axhline(color='k')
            axv = self.canv1.axes.axvline(color='k')
            axh.remove()
            axv.remove()
            self.canv1.draw()
            self.canv1.fig.canvas.mpl_disconnect(self.onClicked)
            self.canv1.fig.canvas.mpl_disconnect(self.mouseMove)

    ### MCA PROCESS ###
    def processMCA(self): 
        global scatterMCA
        if self.select_MCA.text() == "Select MCA" and self.select_MCV.text() == "Select MCV" and self.getSelectGrap == 1:
            self.select_MCA.setText("Deselect MCA")
            self.select_MCA.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
            if self.otomaticGrapMCA != 0:
                self.otomaticGrapMCA.remove()
                self.otomaticGrapMCA = 0
                self.canv2.draw()      
            if scatterMCA != 0:
                scatterMCA.remove()
                scatterMCA = 0
                self.canv2.draw()

            def mouse_move(event):
                if not event.inaxes: # klo di luar axes
                    return

                x, y = event.xdata, event.ydata
                indx = min(np.searchsorted(self.x_filt, x), len(self.x_filt) - 1) # data x diurutin, supaya cursor ngikut sumbu x
                if indx < self.hitungIndex:
                    x = self.x_filt[self.hitungIndex]
                    y = self.y_a_filt[self.hitungIndex]
                else:
                    x = self.x_filt[indx]
                    y = self.y_a_filt[indx]
                axh = self.canv2.axes.axhline(color='k')
                axv = self.canv2.axes.axvline(color='k')
                axh.set_ydata(y)
                axv.set_xdata(x)
                self.canv2.draw()
                axh.remove()
                axv.remove()

            def onclick(event):
                global scatterMCA
                if scatterMCA != 0:
                    scatterMCA.remove()
                x, y = event.xdata, event.ydata
                indx = min(np.searchsorted(self.x_filt, x), len(self.x_filt) - 1) # data x diurutin, supaya cursor ngikut sumbu x
                if indx < self.hitungIndex:
                    x = self.x_filt[self.hitungIndex]
                    y = self.y_a_filt[self.hitungIndex]
                else:
                    x = self.x_filt[indx]
                    y = self.y_a_filt[indx]
                mca = y - self.mcamin_y
                self.hasil_mca.setText(str('%.3f'%(mca)))
                scatterMCA = self.canv2.axes.scatter(x, y, c='black')
                self.canv2.draw()
            
            self.onClicked = self.canv2.fig.canvas.mpl_connect('button_press_event', onclick)
            self.mouseMove = self.canv2.fig.canvas.mpl_connect('motion_notify_event', mouse_move)

        elif self.select_MCA.text() == "Deselect MCA":
            self.select_MCA.setText("Select MCA")
            self.select_MCA.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;} QPushButton::hover{background-color: rgb(57, 57, 63);}")
            axh = self.canv2.axes.axhline(color='k')
            axv = self.canv2.axes.axvline(color='k')
            axh.remove()
            axv.remove()
            self.canv2.draw()
            self.canv2.fig.canvas.mpl_disconnect(self.onClicked)
            self.canv2.fig.canvas.mpl_disconnect(self.mouseMove)

## Save Setting
    def saveSetting(self):
        #### Timer LED #####
        self.duration = self.set_duration.value()
        self.set_white_led_on.setMaximum(self.duration)
        self.set_white_led_off.setMaximum(self.duration)
        self.whiteLedON = self.set_white_led_on.value()
        self.whiteLedOFF = self.set_white_led_off.value()

        #### Setting Plot Graph #####
        self.avgDiameter = self.set_avg_diameter.value()
        self.avgVelocity = self.set_avg_velocity.value()
        self.avgAcceleration = self.set_avg_acceleration.value()

        #### Setting Video #####
        self.fps_in_recording = int(self.set_fps_recording.currentText())
        self.fps_in_display = int(self.set_fps_display.currentText())

        actualFrameCalc = self.duration*self.fps_in_recording
        self.actual_frame.setText(str(actualFrameCalc))
        self.total_frame.setText("0")
        calculation_display1 = 1/self.fps_in_display
        self.actualFPSDisplay = calculation_display1/(1/self.fps_in_recording)
        self.actualFPSDisplay = int(self.actualFPSDisplay)

        dataSetting = ET.Element('Setting')

        #### Timer LED #####
        s_elemSetting1 = ET.SubElement(dataSetting, 'Duration')
        s_elemSetting2 = ET.SubElement(dataSetting, 'WhiteLedON')
        s_elemSetting3 = ET.SubElement(dataSetting, 'WhiteLedOFF')
        s_elemSetting4 = ET.SubElement(dataSetting, 'PortDevice')

        #### Setting Plot Graph #####
        s_elemSetting5 = ET.SubElement(dataSetting, 'AvgDiameter')
        s_elemSetting6 = ET.SubElement(dataSetting, 'AvgVelocity')
        s_elemSetting7 = ET.SubElement(dataSetting, 'AvgAcceleration')

        #### Setting Video #####
        s_elemSetting8 = ET.SubElement(dataSetting, 'FPSinRecording')
        s_elemSetting9 = ET.SubElement(dataSetting, 'FPSinDisplay')

        #### Timer LED #####
        s_elemSetting1.text = str(self.set_duration.value())
        s_elemSetting2.text = str(self.set_white_led_on.value())
        s_elemSetting3.text = str(self.set_white_led_off.value())
        s_elemSetting4.text = self.set_port_device.currentText()

        #### Setting Plot Graph #####
        s_elemSetting5.text = str(self.set_avg_diameter.value())
        s_elemSetting6.text = str(self.set_avg_velocity.value())
        s_elemSetting7.text = str(self.set_avg_acceleration.value())

        #### Setting Video #####
        s_elemSetting8.text = self.set_fps_recording.currentText()
        s_elemSetting9.text = self.set_fps_display.currentText()

        setting_xml = ET.tostring(dataSetting)
        
        with open('setting.xml', "wb") as f:
            f.write(setting_xml)

## Default Setting
    def defaultSetting(self):
        #### Timer LED #####
        self.set_duration.setValue(10)
        self.set_white_led_on.setValue(2)
        self.set_white_led_off.setValue(8)

        #### Setting Plot Graph #####
        self.set_avg_diameter.setValue(5)
        self.set_avg_velocity.setValue(2)
        self.set_avg_acceleration.setValue(2)

        #### Setting Video #####
        self.set_fps_recording.setCurrentText("30")
        self.set_fps_display.setCurrentText("5")

        #### Port Device ####
        self.set_port_device.setCurrentText("COM1")

## Hide Setting
    def hideSetting(self):
        ### Button Setting ###
        self.btn_connecting.setEnabled(False)
        self.btn_connecting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
        self.btn_default_setting.setEnabled(False)
        self.btn_default_setting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
        self.btn_save_setting.setEnabled(False)
        self.btn_save_setting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(66, 66, 66);border-radius:10px;}")
        
        ### Timer LED ###
        self.set_duration.setEnabled(False)
        self.set_white_led_on.setEnabled(False)
        self.set_white_led_off.setEnabled(False)

        ### Port Device ###
        self.set_port_device.setEnabled(False)

        ### Setting Plot Graph ###
        self.set_avg_diameter.setEnabled(False)
        self.set_avg_velocity.setEnabled(False)
        self.set_avg_acceleration.setEnabled(False)

        ### Setting Recording ###
        self.set_fps_recording.setEnabled(False)
        self.set_fps_display.setEnabled(False)

## Open Setting
    def openSetting(self):
        ### Button Setting ###
        self.btn_connecting.setEnabled(True)
        self.btn_connecting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        self.btn_default_setting.setEnabled(True)
        self.btn_default_setting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        self.btn_save_setting.setEnabled(True)
        self.btn_save_setting.setStyleSheet("QPushButton{color: rgb(255, 255, 255);background-color: rgb(31, 32, 39);border-radius:10px;}QPushButton::hover{background-color: rgb(57, 57, 63);}")
        
        ### Timer LED ###
        self.set_duration.setEnabled(True)
        self.set_white_led_on.setEnabled(True)
        self.set_white_led_off.setEnabled(True)

        ### Port Device ###
        self.set_port_device.setEnabled(True)

        ### Setting Plot Graph ###
        self.set_avg_diameter.setEnabled(True)
        self.set_avg_velocity.setEnabled(True)
        self.set_avg_acceleration.setEnabled(True)

        ### Setting Recording ###
        self.set_fps_recording.setEnabled(True)
        self.set_fps_display.setEnabled(True)

app = QtWidgets.QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())