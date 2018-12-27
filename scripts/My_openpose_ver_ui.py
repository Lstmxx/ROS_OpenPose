#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from openpose import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from draw import *
import openpose_ui
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import *
import cv2
from compute import *
g_image = False
d_image = None
openpose = None

def init_openpose():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x320"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["hand_net_resolution"] = "368x368" 
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] =  "/home/negispringfield/openpose/models/"
    openpose = OpenPose(params)
    return openpose

class MainWindow(QtGui.QWidget,openpose_ui.Ui_Form):
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
        self.setupUi(self)
        self.start_flag = False
        self.ishand = False
        self.isbox = True
        self.keypoints = None
        self.timer = QTimer()
        self.timer.start()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.catch_picture)
        # with open('/home/negispringfield/ros_ws/src/python_moveit_tur/scripts/Selection_001.png','rb') as f:
        #     img = f.read()
        # image = QtGui.QImage.fromData(img)
        # pixmap = QtGui.QPixmap.fromImage(image)
        # self.label.setPixmap(pixmap)
        # self.label.setAutoFillBackground(True)
        # self.w = 1107.0
        # self.h = 704
        # self.right_points = []
        # pe = QtGui.QPalette()
        # pe.setColor(QtGui.QPalette.Window,QtGui.QColor(1,84,164,255))
        # self.label.setPalette(pe)
        self.timer_2 = QTimer()
        self.timer_2.start()
        self.timer_2.setInterval(100)
        self.timer_2.timeout.connect(self.catch_head)

        self.timer_3 = QTimer()
        self.timer_3.start()
        self.timer_3.setInterval(100)
        self.timer_3.timeout.connect(self.compute_right)
        self.connect(self.start,SIGNAL('clicked()'),self.check_and_init)
        self.connect(self.hand_on,SIGNAL('clicked()'),self.is_hand_on)
        self.connect(self.box_on,SIGNAL('clicked()'),self.is_box_on)
        rospy.init_node("test")
        rospy.Subscriber("/kinect2/qhd/image_color",Image,self.get_image)
        self.head_list = [self.head_1,self.head_2,self.head_3,self.head_4,self.head_5,self.head_6]
        global openpose
        openpose = init_openpose()
 
    def check_and_init(self):
        #self.head_1.setText("D")
        if self.start.text()=='start':
            
            self.start_flag = True
            self.start.setText('stop')
        else:
            self.start_flag = False 
            self.start.setText('start')
            #self.catch_head()

    def is_hand_on(self):
        if self.ishand:
            self.ishand = False
            self.hand_on.setText('hand_on')
        else:
            self.ishand = True
            self.hand_on.setText('hand_off')

    def is_box_on(self):
        if self.isbox:
            self.isbox = False
            self.box_on.setText('box_on')
        else:
            self.isbox = True
            self.box_on.setText('box_off')

    def catch_picture(self):
        global d_image
        global g_image
        if g_image and self.start_flag:
            self.d_image = d_image
            self.d_image = self.process_picture(self.d_image)
            
            w = self.d_image.shape[1]
            h = self.d_image.shape[0]
            s = 1+(self.w-w)/w
            print(self.w)
            print(w)
            s = round(s,1)
            print(s)
            w=int(w*s)
            h=int(h*s)
            self.d_image = cv2.resize(self.d_image,(0,0),fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
            
            self.d_image = cv2.cvtColor(self.d_image, cv2.COLOR_BGR2RGB)
            w = self.d_image.shape[1]
            h = self.d_image.shape[0]
            
            h1=self.main_Image.height()
            w1=self.main_Image.width()
            print(h1)
            print(w1)
            if w1!=self.w:
                self.w = w1+0.0
            # w_scale = 1+int(w1-w)/w*0.1
            # scale = w_scale 
            # w=w1
            # h=int(h*w_scale)
            # self.d_image = cv2.resize(self.d_image,(w,h),interpolation=cv2.INTER_CUBIC)
            # w = self.d_image.shape[1]
            # h = self.d_image.shape[0]
            #
            # print(self.main_Image.width())
            # print(self.d_image.shape)
            self.image = QtGui.QImage(self.d_image.data,w,h,QtGui.QImage.Format_RGB888)
            
            self.main_Image.setPixmap(QtGui.QPixmap.fromImage(self.image))

    def catch_head(self):
        global d_image
        global g_image
        if g_image and self.start_flag:
            if self.keypoints.shape[0]>0:
                l = self.keypoints.shape[0] if self.keypoints.shape[0] <6 else 6
                for i in range(0,l):
                    value = self.keypoints[i,:,:2].astype(np.int)
                    head = cut_head_picture(value,d_image)
                    #head = cv2.resize(head,(head.shape[0],head.shape[1]),interpolation=cv2.INTER_CUBIC)
                    head = cv2.cvtColor(head,cv2.COLOR_BGR2RGB)
                    self.head_image = QtGui.QImage(head.data,head.shape[1],head.shape[0],QtGui.QImage.Format_RGB888)
                    self.head_list[i].setPixmap(QtGui.QPixmap.fromImage(self.head_image))
                for i in range(l,6-l):
                    self.head_list[i].clear()

    def process_picture(self,img):
        global openpose
        pro = img
        if self.ishand:
            self.keypoints,right,left= openpose.forward(img,hands=self.ishand)
            pro = draw_hand(right,pro)
            pro = draw_hand(left,pro)
        else:
            self.keypoints = openpose.forward(img,hands=self.ishand)
        if self.isbox:    
            pro = people_box(self.keypoints,pro)
        else:
            pro = process_img(self.keypoints,pro)
            #pro,self.right_points = just_right(self.keypoints,pro)
        return pro
    def compute_right(self):
        if self.right_points[0] != [0,0] and self.right_points[1] != [0,0] and self.right_points[2] != [0,0] and self.right_points[3] != [0,0]:
            self.angel_1 = compute_angle(self.right_points[:4])
            self.angel_2 = compute_angle(self.right_points[1:4])
            print(self.angel_1)
            print(self.angel_2)
    def get_image(self,image):
        global d_image
        bridge = CvBridge()
        d_image = bridge.imgmsg_to_cv2(image,"bgr8")
        
        global g_image
        g_image = True


app = QtGui.QApplication(sys.argv)
S = MainWindow()
S.show()
sys.exit(app.exec_())
