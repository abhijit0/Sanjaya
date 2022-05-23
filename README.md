## Required libraries
```
Tensorflow > 2.x
python > 3.6
opencv-python
cudatoolkit ## Which is suitable for your nvida gpu and tensorflow, python version installed
spacy
scipy
matplotlib
```

## How to Run?
- For Video: place the test video in data/test_videos/ directory and run the following command.
```
python3 main.py --video $video_file_name
```
- For Image: place the test video in data/test_iamges/ directory and run the following command.
```
python3 main.py --image $image_file_name
```


## Overview 
Automatic Video Captioning System. "Sanjaya" a mythological character derived from the Mahabharat, who used to describe the war in kuruskhetra (battlefield) to Dhritarashtra, a blind emperor. Inspired by the character, this proejct is aimed at content based audio captions generation for the videos to aid blind people to understand the content of the video. Currently, The captions are displayed which actually contradicts to the aim of the project. However, the captions will be outputed as speech which will be implemented and will reflect in future commits.

## General Idea:
- Image caption method extended to videos.
- Generally video captioning needs a considerably high amount of data annotated with captions. Preparing and gathering this data is time and storage exhaustive process.
- Training the model using this approach would also be computationally expensive and time consuming.
- On the contrary image caption data is relatively is to obtain and prepare.
- So the main idea is to apply the method for image captioning to videos.

## Backbone: Image Caption module using CNN and RNN
- The backbone of the system is the Image captioning module
- The Image captioning module is developed using Resnet18 (CNN) which acts as feature extractor for images, and LSTMs which are used for language modelling.
- Combining these two components, the resulting networks learns to encode images with their linguistic charecterstics which in our case is caption.

## Data Set
- The dataset is Flickr8k dataset which contains 8000 out of which 6000 images are used for training and the rest for testing and validation.
- The training dataset also contains a caption dictionary where each of images are paired with caption list containing varying number of captions.

## Preprocessing
- The images are preprocessed such that each color channel is zero-centered with respect to the ImageNet dataset, without scaling along with converting from RGB to BGR. This step is applied for both training and testing.
- The captions are preprocessed only in training phase:
    - All the letters are converted to lowercase, special characters are removed
    - Each caption is tokenized and padded with the tokens 'startseq' and 'endseq' for processing for LSTMs.
    - The tokens are replaced with their word embeddings from glove_6B_50d word embeddings.
    - The word embeddings are used instead of one hot encodings or direct indices becuase they encode semantic meanings which result in better accuracy.
    

## Image caption architecture

![SAR jpg](https://user-images.githubusercontent.com/17523822/169815791-65484765-e4c4-480f-8bb9-ccc5659225a6.jpg)
## Training : 
- The input image is fed to feature extractor which produces feature maps
- The feature extractor layer is then connected to dense FC (Fully Connected) layer containing 256 neurons.
- We know that the training data set also contains a dictionary where we have list of captions for each image.
- Each caption is fed as input to the network as sequence along with feature map of the image.
- The LSTM module with 256 units leanrs to predict the next word given the sequence of words before it. Here we are feeding word embeddings for             better performance.
- The outputs of both FC layer of feature extractor and LSTM are added. This layer has relu activation.
- Finally we feed the combined output to another FC layer which will contain n number of neurons where n is the vocabulary size.
- The final FC layer will be having softmax activation.
- The model is trained using categorical crossentropy loss using adam optimizer for 10 epochs.


## Challenges
- If we directly apply Image captioning method for each frame, there will be inconsistancy in the outputs for each frame.
- i.e We get different outputs for each frame, although the scene for consecutive frames is similar



## Methodology
- Image captioning is applied frame by frame.
- Traning is done on Images where the input is image and output is the set of captions for each image. 
- The context from frame to frame is captured by the output of the frame (caption), as the minor scene changes in frames is also reflected in minor caption changes 
- we can smooth out the prediction by calculating the similarity between the caption outputs and keep the captions which are semantically similar.
- e.g Sentences "The cat is walking on the road" and "Cat is running on the street" is semantically similar so we keep either of them. 
- the similar frames output similar captions. The similarity of the captions is calculated through cosine similarity of sentance embeddings of these vectors.
- It makes sense to output a different caption for a frame if there is significant scene change between the consecutive frames.
- This change in scene is also reflected in the output caption as well.

## Algorithm for Video Captioning
1. Decalre an array captions = ['']
2. Decalre temporary variable temp 
3. Define threshold t 
4. while next frame is available:
    1. Predict the caption for frame (caption)
    2. calculate cosine similarity between temp and predicted caption (similarity)
    3. if (similarity <= threshold ):
        - append caption to captions
    4. temp = caption


## Results

### Videos

https://user-images.githubusercontent.com/17523822/169861561-ee37ca53-ffd3-493c-aac6-08dcd4a0ee5f.mp4



https://user-images.githubusercontent.com/17523822/169861570-8f19d726-53cb-48bd-b630-dc1cdcdca3c1.mp4



https://user-images.githubusercontent.com/17523822/169861573-fa1da031-2b51-43ba-8dfc-096ce318a28d.mp4


https://user-images.githubusercontent.com/17523822/169295455-95c1b282-65b1-434c-8b51-70260855daaf.mp4


https://user-images.githubusercontent.com/17523822/169861574-4392e77a-09e5-49f3-bffc-2a0d0b38f96f.mp4



https://user-images.githubusercontent.com/17523822/169861579-64883620-b819-4650-a157-e3dc1b0ff138.mp4

---
### Images
![109738763_90541ef30d](https://user-images.githubusercontent.com/17523822/169860813-0496e527-dbae-4dfc-8083-eeb60a9a3321.jpg)
![109738916_236dc456ac](https://user-images.githubusercontent.com/17523822/169860817-8f0fff86-6e3c-4dca-8996-f950cda54839.jpg)
![109823394_83fcb735e1](https://user-images.githubusercontent.com/17523822/169860819-f5f80ffb-aa06-4127-a618-e76c36224be7.jpg)
![im1](https://user-images.githubusercontent.com/17523822/169860821-4a8b22ee-60b7-47be-b32a-c0c82d624090.jpg)

## Conclusion and Future Work
- Although results are not entirely accurate, they are semantically somehow closer to actual depiction of scenes
- The model is relatively simple (Single LSTM layer with 256 units) for language modelling also the dataset is not sufficient enough to cover all types of scenarios
- Aim of the project was to formulate a general idea on how a video captioning bot can be built using already available Image captioning models for efficiency.
- However, there is always room for improvement. I will be continously working on imrpoving the bot. I also welcome anyone interested to contribute to this project :) 

