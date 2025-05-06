Tuberculosis Detection from Chest X-Rays using EfficientNetB0

================================================================

  

This project demonstrates a deep learning-based approach for detecting Tuberculosis (TB) from chest X-ray images. It uses PyTorch and EfficientNetB0 for binary classification (TB vs Non-TB) and includes training, prediction, and evaluation pipelines.

  

Features

----------

  

*  *Binary Classification*: Detects whether a chest X-ray indicates TB or not.

*  *EfficientNetB0 Backbone*: Lightweight and accurate convolutional model.

*  *Image Preprocessing*: Includes resizing, normalization, and augmentation.

*  *Evaluation Metrics*: Generates classification report and confusion matrix.

*  *Predict on Single Image*: Accepts one image input for prediction with confidence score.

  

Prerequisites

----------------

  

* Python 3.7 or later

* PyTorch

* Torchvision

* scikit-learn

* matplotlib

* seaborn

* tqdm

* PIL

  

###  Install dependencies:

  

` pip install torch torchvision scikit-learn matplotlib seaborn tqdm Pillow `

  

Dataset Structure

--------------------

  

Organize your dataset as follows:

  

 tb_dataset/<BR>
    ├── train/<BR>
    │   ├── TB/<BR>
    │   └── Non-TB/<BR>
    ├── val/<BR>
    │   ├── TB/<BR>
    │   └── Non-TB/<BR>
    └── test/<BR>
    .           ├── TB/<BR>
    .           └── Non-TB/<BR>

  

Training

---------------

  

Train the classifier using:

  

`python train_model.py `

  

This will train the model for 10 epochs and save the weights to efficientnet\_tb\_classifier.pth.

  

Evaluation

-------------

  

Run evaluation on the test set using:

  

`python test.py `

  

You will see:

  

* Confusion Matrix

* Classification Report (Precision, Recall, F1)

* Seaborn heatmap of predictions

  

Prediction on Single X-ray Image

-----------------------------------

  

Run prediction on a new image:

  

` python predict.py `

  

You will be prompted:

  

` Please enter the path to the X-ray image: ./samples/tb_xray.jpg Prediction: TB (Confidence: 0.9621) `

  

File Overview

----------------

  

* preprocess.py: Handles image transformations and loads train/val/test DataLoaders

* train\_model.py: Trains EfficientNetB0 on the dataset and saves the model

* predict.py: Loads trained model and makes predictions on a single image

* test.py: Evaluates model performance on the test set using classification metrics

  

Notes

--------

  

* Images should be high-quality X-rays with visible lung regions

* TB images should have distinct pathological regions visible for best accuracy

* You may use any open-source dataset

  

Future Improvements

----------------------

  

* Add Grad-CAM visualization for TB patch highlighting

* Extend to multi-label classification (e.g. Pneumonia, COVID-19)

* Add GUI support or web interface for ease of use
