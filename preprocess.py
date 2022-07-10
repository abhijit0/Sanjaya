#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 22:42:55 2022

@author: a8hik
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
import collections
from time import time
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers import add
        
def load_captions(captions_path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt'):
    with open(captions_path, 'r') as f:
        captions = f.read()
    return captions

def get_captions(captions_path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt'):
    captions = load_captions(captions_path)
    captions = captions.split('\n')[:-1]
    return captions
    
def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-z]+"," ",text)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    text = 'startseq '+text+' endseq'
    return text
    
def get_image_map(path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt'):
    captions = get_captions(captions_path = path)
    descriptions = {}
    for caption in captions:
        caption_tokens = caption.split('\t')
        key = caption_tokens[0][:-2]
        if(key in descriptions.keys()):
            descriptions[key].append(clean_text(''.join(caption_tokens[1:])))
        else:
            descriptions[key] = []
            descriptions[key].append(clean_text(''.join(caption_tokens[1:])))
    descriptions_json = json.loads(str(descriptions).replace("'", "\""))
    return descriptions_json
    
def get_vocab(path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt', threshold = 10):
    descriptions_json = get_image_map(path = path)
    all_vocab = [ word for key in descriptions_json.keys() for description in descriptions_json[key] for word in description.split()]
    counter = collections.Counter(all_vocab)
    sorted_counter = sorted(counter.items(), reverse=True, key = lambda x: x[1])
    sorted_dict = [x for x in sorted_counter if x[1] > threshold]
    vocab = [x[0] for x in sorted_dict]
    #vocab = set([key for key in dict(sorted_counter).keys() if dict(sorted_counter)[key] > threshold])
    return vocab

def preprocess_image(img_path = None, target_size = (224,224,3), input_image=None):
    if(input_image is None):
        img = image.load_img(img_path, target_size = target_size)
    else:
        img = input_image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img     

def ResnetFeatureMaps(weights = 'imagenet', input_shape = (224,224,3), verbose = 0):
    model = ResNet50(weights = weights, input_shape = input_shape)
    model = Model(model.input, model.layers[-2].output)
    return model

def get_encodings(model, img):
    encoded_img = model.predict(img)
    encoded_img = encoded_img.reshape(encoded_img.shape[1])
    return encoded_img
    
    
def train_test_data(dataset_type = 'train', pkl_file_path = None, image_path = './data/Flickr_Dataset/images/' , desc_path = './data/Flickr_Dataset/Flickr_TextData/Flickr_8k.trainImages.txt'):
    descriptions_json = get_image_map()
    map_text = {}
    map_image ={}
    count = 0
    model = ResnetFeatureMaps()
    with open(desc_path, 'r') as f:
        images = f.read().split("\n")[:-1]
    if(pkl_file_path is None or pkl_file_path not in os.listdir(os.getcwd())):
        for image in images:
            img_path = image_path + image
            img = preprocess_image(img_path = img_path)
            encoded_image = get_encodings(model, img)
            map_image[image] = encoded_image
            count += 1
        if(pkl_file_path is None):
            if(dataset_type.lower() == 'train'):
                pkl_file_path = 'train_images_encodings.pkl'
            else:
                pkl_file_path = 'test_images_encodings.pkl'
            with open(pkl_file_path, 'wb') as pickle_file:
                pickle.dump(map_image, pickle_file)
    else:
        with open(pkl_file_path, 'rb') as pkl_file:
            map_image = pickle.load(pkl_file)
            
    if(dataset_type.lower() == 'train'):
        for image in images:
            map_text[image] = descriptions_json[image]
        return json.loads(str(map_text).replace("'", "\"")), map_image
    else:
        return map_image
    
def train_test_data_wfe(dataset_type = 'train', pkl_file_path = None, image_path = './data/Flickr_Dataset/images/' , desc_path = './data/Flickr_Dataset/Flickr_TextData/Flickr_8k.trainImages.txt'):
    descriptions_json = get_image_map()
    map_text = {}
    map_image ={}
    count = 0
    model = ResnetFeatureMaps()
    with open(desc_path, 'r') as f:
        images = f.read().split("\n")[:-1]
    if(pkl_file_path is None or pkl_file_path not in os.listdir(os.getcwd())):
        for image in images:
            img_path = image_path + image
            img = preprocess_image(img_path = img_path)
            img = img.reshape(img.shape[1], img.shape[2], img.shape[3])
            map_image[image] = img
            count += 1
        if(pkl_file_path is None):
            if(dataset_type.lower() == 'train'):
                pkl_file_path = 'train_images_encodings_wfe.pkl'
            else:
                pkl_file_path = 'test_images_encodings_wfe.pkl'
            with open(pkl_file_path, 'wb') as pickle_file:
                pickle.dump(map_image, pickle_file)
    else:
        with open(pkl_file_path, 'rb') as pkl_file:
            map_image = pickle.load(pkl_file)
            
    if(dataset_type.lower() == 'train'):
        for image in images:
            map_text[image] = descriptions_json[image]
        return json.loads(str(map_text).replace("'", "\"")), map_image
    else:
        return map_image

def wti_itw(path = './data/Flickr_Dataset/Flickr_TextData/Flickr8k.token.txt'):
    start_idx = 1
    vocab = get_vocab(path = path)
    word_to_idx = {}
    idx_to_word = {}
    for i in vocab:
        word_to_idx[i] = start_idx
        idx_to_word[start_idx] = i
        start_idx += 1
    vocab_size = len(word_to_idx)
    return word_to_idx ,idx_to_word , vocab_size+1

def max_len(desc = None):
    max_len = max([len(cap.split()) for key in desc.keys() for cap in desc[key]])
    return max_len

def generator(desc, w_ids, vocab_size, image_map, max_len, batch_size):
    images, sequences, y = [], [], [] 
    count = 0
    while True:
        for key, caps in desc.items():
            count += 1
            image = image_map[key]
            
            for cap in caps:
                w_id_tokens = [w_ids[token] for token in cap.split() if token in w_ids]
                for i in range(1, len(w_id_tokens)):
                    in_seq = w_id_tokens[0:i]
                    out_seq = w_id_tokens[i]
                    in_seq = pad_sequences([in_seq], maxlen = max_len, value = 0, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    sequences.append(in_seq)
                    y.append(out_seq)
                    images.append(image)
            if(count == batch_size):
                images_np, sequences_np, y_np = np.array(images), np.array(sequences), np.array(y)
                print(images_np.shape)
                yield ([images_np, sequences_np], y_np) 
                images, sequences, y = [], [], [] 
                count = 0
                
def generator_wfe(desc, w_ids, vocab_size, image_map, max_len, batch_size):
    images, sequences, y = [], [], []
    count = 0
    while True:
        for key, caps in desc.items():
            count += 1
            image = image_map[key]
            
            for cap in caps:
                w_id_tokens = [w_ids[token] for token in cap.split() if token in w_ids]
                for i in range(1, len(w_id_tokens)):
                    in_seq = w_id_tokens[0:i]
                    out_seq = w_id_tokens[i]
                    in_seq = pad_sequences([in_seq], maxlen = max_len, value = 0, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    sequences.append(in_seq)
                    y.append(out_seq)
                    images.append(image)
            if(count == batch_size):
                images_np = np.array(images)
                #images_np = images_np.reshape(images_np.shape[0], images_np.shape[2], images_np.shape[3], images_np.shape[4])
                sequences_np = np.array(sequences)
                y_np = np.array(y)
                print(f'images_np.shape {images_np.shape} {count}')
                yield ([images_np, sequences_np], y_np) 
                images, sequences, y = [], [], [] 
                count = 0

def word_embeddings(vocab_size, w2i, word_embedding_path = 'glove_6B_50d.txt', vec_len = 50):
    word_embedding_map = {}
    with open(word_embedding_path, encoding='utf8') as embeddings:
        for line in embeddings:
            tokens = line.split()
            word_embedding_map[tokens[0]] = np.array(tokens[1:], dtype = 'float')
    
    embedding_matrix = np.zeros((vocab_size, vec_len))
    for word, idx in w2i.items():
        embedding_vec = word_embedding_map.get(word)
        if(embedding_vec is not None):
            embedding_matrix[idx] = embedding_vec
    return embedding_matrix



