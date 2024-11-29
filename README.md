# Smile-Detector
Smile Detector Using InceptionResNetV2

This project aims to develop a model to detect whether a user is smiling based on image input. Below are the detailed steps, dataset, model architecture, and implementation process:

---
## Dataset: GENKI-4K

The dataset consists of 4000 labeled images, categorized as:
- **smile (1)**: Images where the subject is smiling.
- **not-smile (0)**: Images where the subject is not smiling.

---
### Face Cropping
Each image is processed using the `haarcascade_frontalface_default.xml` model from the OpenCV library to detect and crop facial regions. The cropped faces are saved in a new folder named `cropped_pictures` with their respective filenames.

### Label-Based Categorization
Cropped images are organized into two separate folders:
- `0`: For **not-smile** images.
- `1`: For **smile** images.  

This ensures that the `image_dataset_from_directory` function can efficiently read and prepare the images for model training.

---

## Model Architecture

The project uses the **InceptionResNetV2** model from the Keras library. This model is pre-trained on the **ImageNet** dataset, which includes 1000 classes.

### Transfer Learning
The pre-trained weights from ImageNet were leveraged, and the model was fine-tuned to classify the images into two categories: **smile** and **not-smile**.

---

## Model Compilation

The model was compiled with the following parameters:
- **Optimizer**: Adam
- **Loss Function**: `categorical_crossentropy`
- **Metrics**: Accuracy, Precision, and Recall

---

## Model Training

### Training and Validation Split
- **Training Set**: 3600 images.
- **Validation Set**: 400 images.  
Both datasets were normalized using the `preprocess_input` function from `keras.applications.inception_resnet_v2`.

### Training Setup
- **Epochs**: 100
- **EarlyStopping**: Configured with the following parameters:
  - `patience=8`
  - `restore_best_weights=True`
  - `monitor='val_accuracy'`  
This ensures that training halts early if validation accuracy does not improve for 8 consecutive epochs.  

The training was completed in **12 epochs**, thanks to the EarlyStopping callback.

---

## Model Evaluation

The model was evaluated using the validation set (400 images). Metrics such as accuracy, precision, and recall were calculated to assess the model's performance.

---

## Saving the Model

The trained model was saved using the following command for future use:
```python
model.save("Smile_Detector_Model_InceptionResNetV2.keras")

## Real-Time Smile Detection

### Video Input
The OpenCV library and the `VideoCapture` function are used to capture video frames from a webcam.

### Face Detection
Each frame is processed using the `haarcascade_frontalface_default.xml` model to detect and crop facial regions.

### Preprocessing
1. The cropped face is resized to a **256×256** matrix.
2. It is normalized using the `preprocess_input` function from `keras.applications.inception_resnet_v2`.

### Prediction
The preprocessed image is passed through the trained model for prediction using:
```python
model.predict(frame)
