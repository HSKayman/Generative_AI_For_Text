# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:06:49 2024

@author: HSK
"""
import os
import warnings
warnings.filterwarnings('ignore')
for firstNElement in [200,500,1000,3000]:
    for BATCH_SIZE in [64,32,16]:
        for units in [2048,1024,512,256]:
            for EPOCHS in [400,800,1000, 2000]:
                os.system("python ChatBot-PM-Optimization.py {} {} {} {} ".format(firstNElement,BATCH_SIZE,units,EPOCHS))
                print("firstNElement:{}\nBATCH_SIZE: {}\nunits: {}\n EPOCHS:{} is completed.".format(firstNElement,BATCH_SIZE,units,EPOCHS))
