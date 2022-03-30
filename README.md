## Required libraries
```
Tensorflow > 2.x
python > 3.6

```

## How to Run?


## Overview 
Automatic Video Captioning System. "Sanjaya" a character derived from the Mahabharat, who used to describe the war happening in kuruskhetra (battlefield) to Dhritarashtra, a blind emperor. Inspired by the character, this proejct is aimed at content based audio captions generation from videos in real time to aid blind people to understand the content of the video. 

## General Idea:
- Image caption method extended to videos.
- Generally video captioning needs a considerably high amount of data annotated with captions. Preparing and gathering this data is time and storage exhaustive process.
- On the contrary image caption data is relatively is to obtain and prepare.
- So the main idea is to apply the method for image captioning to videos.

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
  - Predict the caption for frame (caption)
  - calculate cosine similarity between temp and predicted caption (similarity)
  - if (similarity <= threshold ):
    - append caption to captions
  - temp = caption

Dataset:
- Flickr8k dataset for training, validation, testing.




## Concolusion

## Results


