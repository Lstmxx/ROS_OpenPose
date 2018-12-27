#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import cv2
import random
import numpy as np
class max_:
    x=0
    y=0
class min_:
    x=0
    y=0
key = ["Nose",
       "Neck",
       "RShoulder",
       "RElbow",
       "RWrist",
       "LShoulder",
       "LElbow",
       "LWrist",
       "MidHip",
       "RHip",
       "RKnee",
       "RAnkle",
       "LHip",
       "LKnee",
       "LAnkle",
       "REye",
       "LEye",
       "REar",
       "LEar",
       "LBigToe",
       "LSmallToe",
       "LHeel",
       "RBigToe",
       "RSmallToe",
       "RHeel",
       "Background"]
color = [x for x in range(0,255)]
def just_right(keypoints,img):
    num=keypoints.shape[0]
    for i in range(0,num):
        value = keypoints[i,1:5,:2].astype(np.int)
        img = draw_py_points(value,key,img)
    return img,value
def key_points(key,points):
    keypoints = dict(zip(key,points))
    return keypoints
def process_img(keypoints,img):
    num=keypoints.shape[0]
    for i in range(0,num):
        value = keypoints[i,:,:2].astype(np.int)
        kp = key_points(key,value)
        img = draw_py_points(value,key,img)
    img = cv2.putText(img,str(num),(10,500),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1)
    return img
def draw_py_points(points,key,img):
    l = len(points)
    
    for i in range(1,l):
        if points[i][0]!=0 and points[i][1]!=0:
            if key[i]=="LShoulder" or key[i]=="MidHip":
                f=points[1]
            elif key[i]=="LHip":
                f=points[8]
            elif key[i]=="REar":
                f=points[i-2]
            elif key[i]=="LEar":
                f=points[i-2]
            elif key[i]=="REye" or key[i]=="LEye":
                f=points[0]
            elif key[i]=="LBigToe":
                f=points[14]
            elif key[i]=="RBigToe":
                f=points[11]
            else:
                f = points[i-1]
            if f[0]==0 and f[1]==0:
                continue
            b = points[i]
            c=random.sample(color,3)
            img = choose_color_and_write(f,b,img)
            #img = cv2.putText(img,key[i],(b[0],b[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1)  
    return img

def draw_hand(handpoints,img):
    if handpoints.shape[0]!=0 and handpoints.shape[1]!=0 and handpoints.shape[2]!=0:
        handpoints_sum = handpoints.shape[0]
        for j in range(0,handpoints_sum):
            hand_point=handpoints[j,:,:2]
            for i in range(1,21):
                if i % 4 == 0:
                    f=hand_point[0]
                else:
                    f=hand_point[i-1]
                    b=hand_point[i]
                if f[0]==0 and f[1]==0:
                    continue
                img = choose_color_and_write(f,b,img,hand=True)
    return img
def choose_color_and_write(f_point,b_point,img,hand=False):
    global color
    c=random.sample(color,3)
    if hand == False:
        img = cv2.circle(img,(f_point[0],f_point[1]),3,c,2,4)
        img = cv2.circle(img,(b_point[0],b_point[1]),5,c,2,0)
        img = cv2.line(img,(f_point[0],f_point[1]),(b_point[0],b_point[1]),c,3)
    img = cv2.line(img,(f_point[0],f_point[1]),(b_point[0],b_point[1]),c,2)
    return img
      
def people_box(keypoints,img):
    img = cv2.putText(img,str(keypoints.shape[0]),(10,500),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1)
    max_x=img.shape[0]
    max_y=img.shape[1]
    num=keypoints.shape[0]
    for i in range(0,num):
        value = keypoints[i,:,:2].astype(np.int)
    # rect = cv2.minAreaRect(value)
        print(value)
        value = filter(lambda x:x[0]!=0 and x[1]!=0,value)
        value = np.array(value,dtype=int)
        x = list(value[:,0])
        y = list(value[:,1])
        # max_.x=(max(x)+10) if (max(x)+10)<=max_x else max_x
        # max_.y=(max(y)+10) if (max(y)+10)<=max_y else max_y
        # min_.x=(min(x)-10) if (min(x)-10)>=0 else 0
        # min_.y=(min(y)-10) if (min(y)-10)>=0 else 0
        max_.x=max(x)+10
        max_.y=max(y)+10
        min_.x=min(x)-10
        min_.y=min(y)-10
        img = cv2.rectangle(img,(max_.x,max_.y),(min_.x,min_.y),(255,0,0),3)
    return img

def cut_head_picture(keypoints,img):
    keypoints = key_points(key,keypoints)
    print(img.shape)
    pro = np.zeros((img.shape))
    if keypoints['Nose'][0]!=0 and keypoints['Nose'][1]!=0:
        x = keypoints['Nose'][0]
        y = keypoints['Nose'][1]
        pro = img[y-30:y+30,x-30:x+30,:]
    return pro