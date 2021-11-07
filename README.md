# Capstone: Detecting Emotions in a Remote Work Era using Facial Emotion Detection

## Problem Statement
Reduction in gatherings and social events, and the increase in remote meetings and work has impacted us, both positively and negatively. The reduced socialising and physical meetings may have affected the way how we're able to read body languages and actions during conversations. Additionally, everyone may have personal problems or issues in life outside of videos that we do not see or notice, but it may affect them in the work they do, or the changes in the facial expressions they exhibit.

- Using facial recognition, can we identify facial expressions and detect emotions?

## Contents
Part 1: Data Cleaning
* Loading of FER2013 and FER+ Datasets
* Data Cleaning and Merging
* Conversion of pixels to images, to save in associated labelled folders

Part 2: Modelling
* Train, Test, Split
* Early Stopping, Checkpoints and ReduceLR
* Modelling
* Fitting
* Misclassification Analysis
* Model Evaluation
* Testing with Random Images
* Conclusion and Further Improvements

## Background

## Datasets
2 datasets have been used for this project as shown below:

 [`fer2013.csv`](./data/fer2013.csv): Facial Emotion Recognition 2013 dataset consisting of approximately 30,000 grayscale facial images of different expressions limited to 48 * 48 pixels. Each image is classified into 7 types of emotion:
* 0 = Angry
* 1 = Disgust
* 2 = Fear
* 3 = Happy
* 4 = Sad
* 5 = Surprise
* 6 = Neutral
[`fer2013plus.csv`](./data/fer2013plus.csv): FER+ dataset consisting of relabelled FER emotions in a probability distribution format per face. The index and rows are exactly the same as the original FER2013 dataset. FER+ introduces 3 additional emotion labels, `contempt`, `NF` and `unknown`.

## Data Dictionary

## Modelling

## Misclassification Analysis

## Evaluation

## Conclusion
While testing with identifying certain emotions, emotions such as contempt and disgust are still difficult to classify. This could likely be due to the lack of training data for those emotions.

Better qualities and larger quantities of images could also aid in clearly segregating the different emotions. There are images within the training dataset that has watermarks. These watermarks lighten/darken certain regions of images where key facial features of emotions are important in their classification.

The dataset used in FER is primarily grayscale, but our model is still able correctly identify the correct emotion on coloured images as they are preprocessed within our function to grayscale images. Certain features which may be minute, but will provide distinctive features for the models to train against. However, coloured images may take longer to train as they have higher channels.

In terms of modelling, while it may not necessarily guarantee higher accuracy, there are definitely more pre-trained models that can be used here, such as AlexNet.

There are also deeper facial features that can be focused on. Certain facial detection projects look specifically deeper into eyes, forehead, mouth, nose, to identify areas of interest for certain emotion classes.

## Further Improvements
Ideally, this project can still be further expanded for more classes of emotions, provided there are images to train. As this project was initially conceived based on wanting to understand people's emotions due to lifestyle and occupational transitions of remote work due to COVID-19, it can be redesigned to incorporate speech recognition to identify tonal changes in conversations to further improve identifying changes in an individual's expressions and feelings during video and/or voice calls.