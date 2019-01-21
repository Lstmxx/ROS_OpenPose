#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_msgs.msg import String
import tf
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
import intera_interface
import math
import numpy as np

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
        self.istrack = False
        self.right_points = []
        self.timer = QTimer()
        self.timer.start()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.catch_picture)
        self.j_dd = 90
        self.move = 'OK'
        with open('/home/negispringfield/ros_ws/src/python_moveit_tur/scripts/Selection_001.png','rb') as f:
            img = f.read()
        image = QtGui.QImage.fromData(img)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.label.setAutoFillBackground(True)
        self.w = 1107.0
        self.h = 704
        self.right_points = []
        pe = QtGui.QPalette()
        pe.setColor(QtGui.QPalette.Window,QtGui.QColor(1,84,164,255))
        self.label.setPalette(pe)

        self.timer_2 = QTimer()
        self.timer_2.start()
        self.timer_2.setInterval(100)
        self.timer_2.timeout.connect(self.catch_head)

        self.timer_3 = QTimer()
        self.timer_3.start()
        self.timer_3.setInterval(100)
        self.timer_3.timeout.connect(self.compute_right)

        self.timer_4 = QTimer()
        self.timer_4.start()
        self.timer_4.setInterval(200)
        self.timer_4.timeout.connect(self.track_arm)

        # self.timer_5 = QTimer()
        # self.timer_5.start()
        # self.timer_5.setInterval(300)
        # self.timer_5.timeout.connect(self.move_arm)

        self.connect(self.start,SIGNAL('clicked()'),self.check_and_init)
        self.connect(self.hand_on,SIGNAL('clicked()'),self.is_hand_on)
        self.connect(self.box_on,SIGNAL('clicked()'),self.is_box_on)
        self.connect(self.track_button,SIGNAL('clicked()'),self.is_track)
        self.label_4.setText(u'当前角度为：')
        rospy.init_node("openpose_ui_ver")

        self.pub = rospy.Publisher('openpose_move',Float64,queue_size=10)
        rospy.Subscriber("/kinect2/qhd/image_color",Image,self.get_image)
        rospy.Subscriber('IS_MOVE', String, self.call_back)
        self.head_list = [self.head_1,self.head_2,self.head_3,self.head_4,self.head_5,self.head_6]
        global openpose
        openpose = init_openpose()


    def is_track(self):
        # starting_joint_angles = {'right_j0': 0,
        #                          'right_j1': 0,
        #                          'right_j2': 0,
        #                          'right_j3': 0,
        #                          'right_j4': 0,
        #                          'right_j5': 0,
        #                          'right_j6': 0}
        # init_pos.init_pos(starting_joint_angles)
        self.pub.publish(0)
        if self.track_button.text() == 'track_on':
            self.istrack = True
            self.track_button.setText('track_off')
            self.move = False
        else:
            self.istrack = False
            self.track_button.setText('track_on')
            self.move = True
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
            # print(self.w)
            # print(w)
            s = round(s,1)
            # print(s)
            w=int(w*s)
            h=int(h*s)
            self.d_image = cv2.resize(self.d_image,(0,0),fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
            
            self.d_image = cv2.cvtColor(self.d_image, cv2.COLOR_BGR2RGB)
            w = self.d_image.shape[1]
            h = self.d_image.shape[0]
            
            h1=self.main_Image.height()
            w1=self.main_Image.width()
            # print(h1)
            # print(w1)
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
            self.keypoints,self.right,self.left= openpose.forward(img,hands=self.ishand)
            pro = draw_hand(self.right,pro)
            pro = draw_hand(self.left,pro)
        else:
            self.keypoints = openpose.forward(img,hands=self.ishand)
        if self.isbox:    
            pro = people_box(self.keypoints,pro)
        elif self.istrack:
            pro,self.right_points = just_right(self.keypoints,pro)
        elif self.istrack!=True and self.isbox!=True:
            pro = process_img(self.keypoints,pro)
            #pro,self.right_points = just_right(self.keypoints,pro)
        return pro
    def compute_right(self):
        # if self.right_points[0] != [0,0] and self.right_points[1] != [0,0] and self.right_points[2] != [0,0] and self.right_points[3] != [0,0]:
        #     self.angel_1 = compute_angle(self.right_points[:4])
        #     self.angel_2 = compute_angle(self.right_points[1:4])
        #     print(self.angel_1)
        #     print(self.angel_2)
        if self.ishand and (self.right !=[] or self.right != None):
            hand_point=self.right[0,5:9,:2]
            flag = False
            for f in hand_point:
                if f[0]==0 and f[1] == 0:
                    flag = False
                    break
                else:
                    flag = True
            if flag:
                k_1 = (hand_point[0][1]*1.0-hand_point[1][1])/(hand_point[0][0]-hand_point[1][0])
                k_2 = (hand_point[1][1]*1.0-hand_point[3][1])/(hand_point[1][0]-hand_point[3][0])
                if abs(k_1 - k_2)<3:
                    print('1')
            
    def track_arm(self):
        if self.istrack:
            if self.right_points!=[]:
                #print(len(self.right_points))
                if len(self.right_points) == 5 and self.move == 'OK':
                    k1 = self.compute_k(self.right_points[0],self.right_points[-1])
                    k2 = self.compute_k(self.right_points[1],self.right_points[2])
                    j_d = int(math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5)
                    if self.right_points[2][1]<self.right_points[0][1] and j_d<80:
                        j_d = 90 - j_d + 90
                    if abs(j_d-self.j_dd)>10:
                        self.j_dd = j_d
                        self.r =j_d/90.0 - 1
                        self.label_4.setText(u'当前角度为：'+str(self.j_dd))
                        print(self.r)
                        if abs(self.r)<=1:
                            self.pub.publish(self.r)
                            
    # def move_arm(self):
    #     if self.move == True and self.j<10:
    #         starting_joint_angles = {'right_j0': 0,
    #                              'right_j1': -self.i,
    #                              'right_j2': 0,
    #                              'right_j3': 0,
    #                              'right_j4': 0,
    #                              'right_j5': 0,
    #                              'right_j6': 0}
    #         init_pos.init_pos(starting_joint_angles)
    #         self.move = False
    def get_image(self,image):
        global d_image
        bridge = CvBridge()
        d_image = bridge.imgmsg_to_cv2(image,"bgr8")
        global g_image
        g_image = True
    def compute_k(self,point_1,point_2):
        k = (point_1[1]*1.0-point_2[1])/(point_1[0]-point_2[0])
        return k
    def call_back(self,data):
        self.move = data.data
app = QtGui.QApplication(sys.argv)
S = MainWindow()
S.show()
sys.exit(app.exec_())
