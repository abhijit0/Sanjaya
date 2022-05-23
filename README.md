## Required libraries
```
Tensorflow > 2.x
python > 3.6

```

## How to Run?
- For Video
```
python3 main.py --video test4.mp4
```
- For Image 
```
python3 main.py --video test4.mp4
```


## Overview 
Automatic Video Captioning System. "Sanjaya" a character derived from the Mahabharat, who used to describe the war happening in kuruskhetra (battlefield) to Dhritarashtra, a blind emperor. Inspired by the character, this proejct is aimed at content based audio captions generation from videos in real time to aid blind people to understand the content of the video. 

## General Idea:
- Image caption method extended to videos.
- Generally video captioning needs a considerably high amount of data annotated with captions. Preparing and gathering this data is time and storage exhaustive process.
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
- The images are preprocessed such that each color channel is zero-centered with respect to the ImageNet dataset, without scaling along with converting from RGB to BGR.
- The captions are preprocessed in many steps namely:
    -  

## Image caption architecture

![SAR jpg](https://user-images.githubusercontent.com/17523822/169815791-65484765-e4c4-480f-8bb9-ccc5659225a6.jpg)

- The input image is fed to feature extractor which produces feature maps
- 


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

Dataset:
- Flickr8k dataset for training, validation, testing.




## Concolusion

## Results


https://user-images.githubusercontent.com/17523822/169295455-95c1b282-65b1-434c-8b51-70260855daaf.mp4



