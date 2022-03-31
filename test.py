#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:31:21 2022

@author: a8hik
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO
import pyglet
from tempfile import TemporaryFile, NamedTemporaryFile
import playsound
import pygame

from preprocess import *
nlp = spacy.load("en_core_web_md")
caption_token_path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt'
pygame.init()

def predict_caption_image(model, w2i, i2w, image, n = 1, maxlen = 35):
    text = 'startseq'
    caption_pred = ''
    for i in range(maxlen):
        sequence = [w2i[token] for token in text.split() if token in w2i.keys()]
        sequence = pad_sequences([sequence], maxlen = maxlen, padding='post')
        output = model.predict([image, sequence])
        output = output.argmax()
        output_word = i2w[output]
        text +=' '+output_word
        if output_word=='endseq':
            break
    return ' '.join([token for token in text.split() if token not in ('startseq', 'endseq')])

'''def predict(model_path = './model_10.h5', data_path = './data/Flickr_Dataset/images/', encoding_path = 'test_images_encodings.pkl', test_size = 10):
    #if(encoding_path not in os.listdir(os.getcwd())):
    model = load_model(model_path)
    w2i, i2w, _ = wti_itw(path = caption_token_path)
    test_images_map = train_test_data(dataset_type='test', pkl_file_path= encoding_path , desc_path='./data/Flickr_Dataset/Flickr_TextData/Flickr_8k.testImages.txt')
    test_images = list(test_images_map.keys())[-test_size:]
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (5, 64)
  
    # fontScale
    fontScale = 0.5
   
    # Red color in BGR
    color = (0, 0, 255)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    
    plot_images = []
    captions = []
    for image in test_images:
        feature_map = test_images_map[image].reshape(1,-1)
        image_plot = cv2.imread(os.path.join(data_path, image))
        image_resized = cv2.resize(image_plot, (512, 256))
        cap_predicted = predict_caption_image(model, w2i, i2w, feature_map)
        cap_array = np.zeros((128,512,3), dtype='uint8')
        cap_image = cv2.putText(cap_array, cap_predicted, org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
        image_final = np.concatenate((image_resized, cap_image))
        #image_final = np.append(image_resized, cap_image)
        plot_images.append(image_final)
        captions.append(cap_predicted)
        
    #cv2.namedWindow('captions', cv2.WINDOW_AUTOSIZE)
    for image in plot_images:
        cv2.imshow('images', image)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()'''

def window_config():
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (5, 64)
  
    # fontScale
    fontScale = 0.5
   
    # Red color in BGR
    color = (0, 0, 255)
  
    thickness = 2
    return font, org, fontScale, color, thickness
   
    
def predict(model_path = './model_10.h5', data_path = './data/Flickr_Dataset/images/', save_path = 'data/results_image/', image_name = None):
    #if(encoding_path not in os.listdir(os.getcwd())):
    resnet_model = ResnetFeatureMaps()
    sequence_model = load_model(model_path)
    
    image = cv2.imread(data_path)
    w2i, i2w, _ = wti_itw(path = caption_token_path)
    
    image_plot = cv2.resize(image, (512, 256))
    image = cv2.resize(image, (224,224))
    preprocessed_frame = preprocess_image(input_image = image)
    frame_encoding = get_encodings(resnet_model, preprocessed_frame).reshape(1,-1)
    cap_predicted = predict_caption_image(sequence_model, w2i, i2w, frame_encoding)    
    cap_array = np.zeros((128,512,3), dtype='uint8')
    font, org, fontScale, color, thickness = window_config()
    cap_image = cv2.putText(cap_array, (cap_predicted), org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    image_final = np.concatenate((image_plot, cap_image))
    
    cv2.imwrite(f'{save_path}/{image_name}',image_final)    
    #cv2.namedWindow('captions', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('images', image_final)
    #cv2.waitKey(10)
    #cv2.destroyAllWindows()
    

def cap_similarity(cap1, cap2):
    cap1_vector = nlp(cap1).vector
    cap2_vector = nlp(cap2).vector
    return cosine_similarity([cap1_vector, cap2_vector])
    
def predict_video(model_path = './model_10.h5', video_path = './data/test_videos/test6.mp4', save_path = 'data/results_video/', video_name= None):
    resnet_model = ResnetFeatureMaps()
    sequence_model = load_model(model_path)
    
    w2i, i2w, _ = wti_itw(path = caption_token_path)
    cap = cv2.VideoCapture(video_path)
    
    font, org, fontScale, color, thickness = window_config()
    captions = []
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    
    threshold = 1.0    
    temp = ''
    frames = []
    captions = ['']
    captions_audio = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret == True): 
            frame_plot = frame
            frame_plot = cv2.resize(frame_plot, (512, 256))
            frame = cv2.resize(frame, (224,224))
            preprocessed_frame = preprocess_image(input_image = frame)
            frame_encoding = get_encodings(resnet_model, preprocessed_frame).reshape(1,-1)
            cap_predicted = predict_caption_image(sequence_model, w2i, i2w, frame_encoding)
            similarity = cap_similarity(temp, cap_predicted)
            
            if(similarity[0][1] <= threshold):
                captions.append(cap_predicted)
            
            cap_array = np.zeros((128,512,3), dtype='uint8')
            captions_audio.append(captions[-1])
            cap_image = cv2.putText(cap_array, (captions[-1]), org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
            image_final = np.concatenate((frame_plot, cap_image))
            frames.append(image_final)
            temp = cap_predicted
        else:
            break
    cap.release()
    
    #print(frames[0].shape)
    #print((frames[0].shape[0],frames[0].shape[1]))
    result = cv2.VideoWriter(f'{save_path}/{video_name[:-4]}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (frames[0].shape[1],frames[0].shape[0]))
    
    for frame in frames:
        #cv2.imshow('Frame', frame)
        result.write(frame)
        cv2.waitKey(30)
    
    '''for caption,frame in zip(captions_audio, frames):
        if(captions_audio.index(caption) > 0 and captions_audio.index(caption) < len(captions_audio)):
            prev_caption = captions_audio[captions_audio.index(caption) - 1]
            if(caption != prev_caption):
                play_audio(caption)
            
        cv2.imshow('Frame', frame)
        cv2.waitKey(50)
            

    cv2.destroyAllWindows()'''

def play_audio(text = 'Hello there'):
    mp3_fp = '/tmp/temp.wav'
    tts = gTTS(text, lang='en')
    tts.save(mp3_fp)
   
    playsound.playsound(mp3_fp)
    os.remove(mp3_fp)
    
