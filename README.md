# DermDetect - A Skin Cancer Detection Model With IOS Integration
## Exposition
Skin cancer is an extremely harmful disease that presents itelf in early stages, often unrecognized by the untrained eye of the patient. In addition, clinical procedures for official diagnosis of skin cancer often take up to a month. In this time the condition may worsen considerably and leaves the patient in a state of panic and concern. In order to present an alternative perspective on patient skin condition that can be accessed conviently and quickly, we have developed DermDetect. DermDetect is an application that harnesses the intellecual power of Residual Network 50 (ResNet50) in combination with the accessibility of IOS applications to provide free skin classification on four core categories thusfar: Benign Lesions of the Keratosis, Melanoma, and Melanocytic Nevi.

The sustainable development goal being tackled here is SDG 3: Good Health and Well-Being.

## Technical Details
DermDetect scans an image taken within the app for a classification of the blemish that worries the user.

The installable application DermDetect should be found on the appstore, installed for free. Usage will require a functioning camera. Simply take a close up picture of the blemish in question and allow the program to determine the rest. 

## Model Details
The model powering every classification within DermDetect is a transfer-learning system utilizing ResNet-50. The model features approximately 23 million parameters.

Several different datasets were explored, but the one yielding best results is an equisize-processed HAM10000 (Humans against machines 10000) dataset. This dataset has seen extensive exploration and is curated by first author Noel Codella. You may read more [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

During training, HAM10000(3)-modelv14 demonstrated to perform not necessarily exceptionally well at approximately 84% validation accuracy and above 95% training accuracy. In my opinion the issue of validation accuracy being less than 90% lies in model architectural limitations, as the loss function (incorporated an early_stop callback function) reported very low scores and less overfitting. 

### Demo Video
[![Demonstration DermDetect](https://i.imgur.com/a/Lo9zWGy)](https://www.youtube.com/watch?v=cDYIFwmEafs&ab)

## Development Overview

### Design Process
There were several key elements within the design of the transfer learning system, prior to the VGG-16 implementation currently underway. An important consideration is data size consistency and processability of data and data volume imbalance calibration.

This model underwent several evolutions of training. First I had explored a different dataset from the HAM10000 from Kaggle. Several datasets passed I had refined and began to utilize different methodologies for our training. I started implementing different variations of earlystop callbacks and checkpoints in order to manage the model's training since it often took hours or even full days.

Future work will include deviating from the current model ResNet and attempting to utilize self-made architecture. Possibly taking the VGG-16 base model and translating it larger or experimenting with it. ResNet's architecture cannot be augmented once implemented so it is unappealing.

## Sources
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
https://huggingface.co/docs/transformers/model_doc/resnet
https://www.mathworks.com/help/deeplearning/ref/vgg16.html
https://www.kaggle.com/code/fanconic/cnn-for-skin-cancer-detection/notebook
https://www.youtube.com/watch?v=qB6h5CohLbs&ab_channel=DigitalSreeni
### ChatGPT