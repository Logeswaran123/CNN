# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:24:27 2019

@author: admin
"""

import os
import cv2
from mss import mss
import numpy as np
import keyboard
from selenium import webdriver
import time


def preprocessing(img):
    img = img[::,75:615]
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    return img

# capture
def start():
    

    sct = mss()

    coordinates = {
        'top': 200,
        'left': 220,
        'width': 1000,
        'height': 230,
    }

    with open('actions.csv', 'w') as csv:

        x = 0
        
        # if no directory. create directory
        if not os.path.exists(r'./images'):
            os.mkdir(r'./images')

        while True:
            img = preprocessing(np.array(sct.grab(coordinates)))
            
            # '1'-Jump, '2'-Duck, '0'-No action. Capture frames and store corresponding key actions
            if keyboard.is_pressed('up arrow'): 
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                csv.write('1\n')
                print('jump write')
                x += 1

            if keyboard.is_pressed('down arrow'):
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                csv.write('2\n')
                print('duck')
                x += 1

            if keyboard.is_pressed('t'):
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                csv.write('0\n')
                print('nothing')
                x += 1

            # break the video feed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                csv.close()
                cv2.destroyAllWindows()
                break

# Chromedriver location
driver = webdriver.Chrome(r"C:\Users\admin\Downloads\chromedriver_win32\chromedriver")

# internet connection must be off
driver.get('http://www.google.com/')
time.sleep(2)
page = driver.find_element_by_class_name('offline')
page.send_keys(u'\ue00d')

start()

while True:
    pass