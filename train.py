#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:31:18 2022

@author: a8hik
"""
from preprocess import *


def build_model_e2e(maxlen = 35, vocab_size = 1848, vec_dim = 50):
    resnet_model = ResnetFeatureMaps()
    intermediate_layer = resnet_model.output
    ie_dropout1 = Dropout(0.3)(intermediate_layer)
    ie_dense = Dense(256, activation= 'relu')(ie_dropout1)
    
    # Building sequence Model
    input_seq = Input(shape=(maxlen, ))
    embedding = Embedding(input_dim=vocab_size, output_dim=vec_dim, mask_zero=True)(input_seq)
    s_dropout1 = Dropout(0.3)(embedding)
    lstm_cell = LSTM(256)(s_dropout1)
    
    #Add the outputs of final layers of feature extractor and sequence model
    combined_rep = add([ie_dense, lstm_cell])
    decoder = Dense(256, activation='relu')(combined_rep)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    merged_model = Model(inputs = [resnet_model.input, input_seq], outputs = outputs)
    
    return merged_model

def build_model(input_feature_shape = (2048, ), maxlen = 35, vocab_size = 1848, vec_dim = 50):
    
    # Budilng Feature extractor
    input_feature = Input(shape=input_feature_shape)
    ie_dropout1 = Dropout(0.3)(input_feature)
    ie_dense = Dense(256, activation= 'relu')(ie_dropout1)
    
    # Building sequence Model
    input_seq = Input(shape=(maxlen, ))
    embedding = Embedding(input_dim=vocab_size, output_dim=vec_dim, mask_zero=True)(input_seq)
    s_dropout1 = Dropout(0.3)(embedding)
    lstm_cell = LSTM(256)(s_dropout1)
    
    #Add the outputs of final layers of feature extractor and sequence model
    combined_rep = add([ie_dense, lstm_cell])
    decoder = Dense(256, activation='relu')(combined_rep)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    
    
    #Merging the two models
    merged_model = Model(inputs = [input_feature, input_seq], outputs = outputs)
    return merged_model

def set_weights(model, word_embeddings):
    model = build_model()
    model.layers[2].set_weights([word_embeddings])
    model.layers[2].trainable = False
    return model

def set_weights_e2e(model, word_embeddings):
    model = build_model_e2e()
    model.layers[-8].set_weights([word_embeddings])
    model.layers[-8].trainable = False
    return model    

def train(epochs = 10, batch_size = 32, optimizer = 'adam', verbose = 0):
    model = build_model()
    w2i, _, vocab_size = wti_itw()
    embeddings = word_embeddings(vocab_size, w2i)
    model = set_weights(model, embeddings)
    model.compile(loss = 'categorical_crossentropy', optimizer  = optimizer)
    desc_map, image_map = train_test_data(dataset_type='train', pkl_file_path= 'encoded_train_images.pkl')
    
    maxlen = max_len(desc_map)
    data_generator = generator(desc_map, w2i, vocab_size, image_map, maxlen, batch_size)
    steps_per_epoch = len(desc_map)//batch_size
    
    model.fit_generator(data_generator, epochs = 10, steps_per_epoch = steps_per_epoch, verbose = verbose)
    model.save(f'model_{epochs}.h5')
    
def train_e2e(epochs = 10, batch_size = 1, optimizer = 'adam', verbose = 0):
    model = build_model_e2e()
    w2i, _, vocab_size = wti_itw()
    embeddings = word_embeddings(vocab_size, w2i)
    model = set_weights_e2e(model, embeddings)
    model.compile(loss = 'categorical_crossentropy', optimizer  = optimizer)
    desc_map, image_map = train_test_data_wfe(dataset_type='train', pkl_file_path='train_images_encodings_wfe.pkl')
    
    maxlen = max_len(desc_map)
    data_generator = generator_wfe(desc_map, w2i, vocab_size, image_map, maxlen, batch_size)
    steps_per_epoch = len(desc_map)//batch_size
    
    model.fit_generator(data_generator, epochs = 10, steps_per_epoch = steps_per_epoch, verbose = verbose)
    model.save(f'model_{epochs}.h5')

   
#model = build_model_e2e()
#print(model.summary())
train(verbose=1)