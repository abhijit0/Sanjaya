#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:48:13 2022

@author: a8hik
"""
import sys
import os
from test import predict, predict_video
from os.path import exists

if __name__== '__main__':
    try:
        if(str(sys.argv[1]) == '--image'):
            file_name = sys.argv[2]
            file_path = os.path.join(os.getcwd(), 'data/test_images/' ,file_name)
            print(file_path)
            if(exists(file_path)):
                predict(data_path=file_path, image_name = file_name)
                print(f'Results have been saved in data/results_image folder')
            else:
                raise FileNotFoundError
        elif(str(sys.argv[1]) == '--video'):
            file_name = sys.argv[2]
            file_path = os.path.join(os.getcwd(), 'data/test_videos/' ,file_name)
            print(file_path)
            if(exists(file_path)):
                predict_video(video_path=file_path, video_name = file_name)
                print(f'Results have been saved in data/results_image folder')
            else:
                raise FileNotFoundError 
    except(FileNotFoundError):
        print(FileNotFoundError)
            
            