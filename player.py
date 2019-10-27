# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:27:06 2019

@author: admin
"""

import selenium
import cv2
import numpy as np
import time
from keras.models import load_model
from selenium import webdriver
from mss import mss

model = load_model('chrome_dinogame.h5')

start = time.time()

def predict(game_element):

    # set coordinates
    sct = mss()
    coordinates = {'top': 200, 'left': 220, 'width': 1000, 'height': 230, }

    # capture the image
    img = np.array(sct.grab(coordinates))

    # crop the area, detect edges and resize
    img = img[::,75:615]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = img[np.newaxis, :, :, np.newaxis]
    img = np.array(img)

    # model prediction
    y_prob = model.predict(img)
    prediction = y_prob.argmax(axis=-1)
    
    # Dino Jump
    if prediction == 1:
        game_element.send_keys(u'\ue013')
        print('Jump')
        time.sleep(.07)
    # do nothing
    if prediction == 0:
        print('Nothing')
        pass
    # Dino Duck
    if prediction == 2:
        print('Duck')
        game_element.send_keys(u'\ue015')

 
# Chrome driver location
driver = webdriver.Chrome(r"C:\Users\admin\Downloads\chromedriver_win32\chromedriver")

# internet connection must be off
driver.get('http://www.google.com/')
time.sleep(2)

# main page to send key commands to
page = driver.find_element_by_class_name('offline')

# start game
page.send_keys(u'\ue00d')

# controls the dinosaur
while True:
   predict(page) 