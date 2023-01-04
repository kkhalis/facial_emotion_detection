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

## Datasets
2 datasets have been used for this project as shown below:

 [`fer2013.csv`](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge): Facial Emotion Recognition 2013 dataset consisting of approximately 30,000 grayscale facial images of different expressions limited to 48 * 48 pixels. Each image is classified into 7 types of emotion:
* 0 = Angry
* 1 = Disgust
* 2 = Fear
* 3 = Happy
* 4 = Sad
* 5 = Surprise
* 6 = Neutral

[`fer2013plus.csv`](https://github.com/microsoft/FERPlus): FER+ dataset consisting of relabelled FER emotions in a probability distribution format per face. The index and rows are exactly the same as the original FER2013 dataset. FER+ introduces 3 additional emotion labels, `contempt`, `NF` and `unknown`.

## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**emotion**|*int64*|fer2013.csv|Category of emotion in FER2013|
|**pixels**|*object*|fer2013.csv|Array of image pixel with brightness values ranging from 0-255|
|**usage**|*object*|fer2013.csv|Intended usage of image between:<br> - Training <br> - PublicSet <br> - PrivateSet |
|**Usage**|*object*|fer2013plus.csv|Intended usage of image between:<br> - Training <br> - PublicSet <br> - PrivateSet |
|**Image name**|*object*|fer2013plus.csv|Specified image name|
|**neutral**|*int64*|fer2013plus.csv|Probability of image being a neutral emotion|
|**happiness**|*int64*|fer2013plus.csv|Probability of image being a happiness emotion|
|**surprise**|*int64*|fer2013plus.csv|Probability of image being a surprise emotion|
|**sadness**|*int64*|fer2013plus.csv|Probability of image being a sadness emotion|
|**anger**|*int64*|fer2013plus.csv|Probability of image being a anger emotion|
|**disgust**|*int64*|fer2013plus.csv|Probability of image being a disgust emotion|
|**fear**|*int64*|fer2013plus.csv|Probability of image being a fear emotion|
|**contempt**|*int64*|fer2013plus.csv|Probability of image being a contempt emotion|
|**unknown**|*int64*|fer2013plus.csv|Probability of image being a unknown emotion|
|**NF**|*int64*|fer2013plus.csv|Probability of image emotion that is not found|


## Misclassification Analysis
Looking at the misclassified data, I noticed that images are commonly misclassifed around anger, happiness or neutral. These misclassified images tend to have the presence of teeth, which can be misleading to the model as a smile due to brighter pixels around the mouth region. Additionally, pouts which have no exposure of teeth, became misclassified as a neutral emotion instead.

Obscured facial features contribute to misclassification mostly due to 2 reasons. The key facial features are covered, so the model has lesser features to make an accurate prediction. Or, the model cannot detect a face, given hands are not part of facial detection here.

## Evaluation
After utilising a few pre-trained models, I found out that Resnet50 managed with the highest accuracy score among the rest. The accuracy also plateaus around the 30th epoch, with minimal or no improvements in accuracy after.

Pointing your attention to freezing of layers, using VGG16 without freezing the layers yielded a higher accuracy score compared to VGG16 with frozen layers. I hypothesize that the weights from imagenet might not cater too well specifically to facial expressions. Additionally, Resnet152 with 152 layers didn't perform as well, showing that models with more layers doesn't necessarily provide better accuracy.

## Conclusion
While testing with identifying certain emotions, emotions such as contempt and disgust are still difficult to classify. This could likely be due to the lack of training data for those emotions.

Better qualities and larger quantities of images could also aid in clearly segregating the different emotions. There are images within the training dataset that has watermarks. These watermarks lighten/darken certain regions of images where key facial features of emotions are important in their classification.

The dataset used in FER is primarily grayscale, but our model is still able correctly identify the correct emotion on coloured images as they are preprocessed within our function to grayscale images. Certain features which may be minute, but will provide distinctive features for the models to train against. However, coloured images may take longer to train as they have higher channels.

In terms of modelling, while it may not necessarily guarantee higher accuracy, there are definitely more pre-trained models that can be used here, such as AlexNet.

There are also deeper facial features that can be focused on. Certain facial detection projects look specifically deeper into eyes, forehead, mouth, nose, to identify areas of interest for certain emotion classes.

## Further Improvements
Ideally, this project can still be further expanded for more classes of emotions, provided there are images to train. As this project was initially conceived based on wanting to understand people's emotions due to lifestyle and occupational transitions of remote work due to COVID-19, it can be redesigned to incorporate speech recognition to identify tonal changes in conversations to further improve identifying changes in an individual's expressions and feelings during video and/or voice calls.

## References
[1] FER Dataset - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

[2] FER+ Dataset - https://github.com/microsoft/FERPlus

[3] From raw images to real-time predictions with Deep Learning - https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
