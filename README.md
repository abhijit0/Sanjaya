## Required libraries
```
Tensorflow > 2.x
python > 3.6

```

## How to Run?


## Overview 
Automatic Video Captioning System. "Sanjaya" a character derived from the Mahabharat, who used to describe the war happening in kuruskhetra (battlefield) to Dhritarashtra, a blind emperor. Inspired by the character, this proejct is aimed at content based audio captions generation from videos in real time to aid blind people to understand the content of the video. 

## General Idea:
- Image caption
- pretrained CNN backend for feature extraction. o/p: Encodings of these images called feature maps
- LSTM for encoding the context of the feature maps and sequences of caption tokens.
- I/O for LSTM is thus image feature map and the token, output is the next token of the caption for the current image.
- 


## Methodology
- Extension of Image captioning on video. Image captioning is applied frame by frame.
- Traning is done on Images
- The context from frame to frame is captured by the output of the frame (caption), and not frames themsleves
- Encoding context from frames is computationally expensive process.
- Meaning, we need to have a video dataset with captions for each frame by video, and need to feed thousands of, if not hundreds of thousands of videos for training
- Instead, we can smooth out the prediction by calculating the similarity between the caption outputs.
- The idea is simple, the similar frames output similar captions. The similarity of the captions is calculated through cosine similarity of sentance embeddings of these vectors.
- It makes sense to output a different caption for a frame if there is significant scene change between the consecutive frames.
- This change in scene is also reflected in the output caption as well.

Dataset:
- Flickr8k dataset for training, validation, testing.




## Concolusion

## Results


