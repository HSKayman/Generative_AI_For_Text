# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:06:49 2024

@author: HSK
"""
import os

for firstNElement in [100,200,500]:
    for BATCH_SIZE in [64,32,16]:
        for units in [1024,512,256]:
            for EPOCHS in [100,300,400,800,1000]:
                os.system("python ChatBot-PM-Optimization.py {} {} {} {} ".format(firstNElement,BATCH_SIZE,units,EPOCHS))