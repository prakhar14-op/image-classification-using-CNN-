# image-classification-using-CNN-
Cat & Dog Image Classification using CNN 
- This project is a classic computer vision problem that uses a Convolutional Neural Network (CNN) to classify images as either containing a cat or a dog. The model is built from scratch using TensorFlow and Keras.
## Overview
The goal of this project is to build an efficient and accurate image classifier. The CNN is trained on a dataset of thousands of cat and dog images to learn the distinguishing features of each animal. The project covers data preprocessing, model building, training, and evaluation.

## 📂 Dataset

The model was trained on the **Dogs vs. Cats** dataset, which was originally a Kaggle competition.

* **Training Set:** Contains thousands of images of cats and dogs for training the model.
* **Test Set:** Contains images for validating the model's performance.

You can download the dataset from [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data). You will need to structure the data into separate `training_set` and `test_set` directories, each with `cats` and `dogs` subdirectories, like this:
```
dataset/
├── training_set/
│   ├── cats/
│   │   ├── cat.1.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog.1.jpg
│       └── ...
└── test_set/
    ├── cats/
    │   ├── cat.4001.jpg
    │   └── ...
    └── dogs/
        ├── dog.4001.jpg
        └── ...
```
## 🧠 Model Architecture

The classifier is a **Sequential Convolutional Neural Network (CNN)** built from scratch using TensorFlow's Keras API. The architecture is designed to effectively learn hierarchical features from the cat and dog images.

The layers are stacked in the following order:

1.  **Convolutional Layer (`Conv2D`):** The first layer uses 32 filters with a (3,3) kernel size and a `relu` activation function. It scans the input images to detect basic features like edges and textures.

2.  **Max Pooling Layer (`MaxPooling2D`):** A pooling layer with a (2,2) pool size is added to downsample the feature maps. This reduces computational complexity and makes the detected features more robust.

3.  **Second Convolutional & Pooling Block:** A second set of `Conv2D` and `MaxPooling2D` layers is added to allow the model to learn more complex and abstract patterns from the features identified by the first block.

4.  **Flattening Layer (`Flatten`):** This layer converts the 2D feature maps from the convolutional blocks into a 1D vector. This is necessary to transition from the convolutional part of the network to the dense, fully connected part.

5.  **Fully Connected Layer (`Dense`):** A hidden dense layer with 128 neurons and `relu` activation. This layer performs classification based on the features extracted by the convolutional layers.

6.  **Output Layer (`Dense`):** The final layer consists of a single neuron with a `sigmoid` activation function. It outputs a probability value between 0 and 1, where a value closer to 0 represents a 'cat' and a value closer to 1 represents a 'dog'.

The model is compiled using the `adam` optimizer and `binary_crossentropy` as the loss function, which is the standard choice for binary (two-class) classification problems. The primary metric tracked during training is `accuracy`.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for training and testing purposes.

### Prerequisites

Make sure you have Python 3.7+ and `pip` installed on your system. You can check your versions with the following commands:

```bash
python --version
pip --version
```
# Installation
   1.  Clone the repository
      - First, clone this repository to your local machine. Replace your-username and your-repo-name with your actual GitHub username and repository name.
       ```
       git clone [https://github.com/prakhar14-op/image-classification-using-CNN-.git](https://github.com/prakhar14-op/image-classification-using-CNN-.git)
       cd image-classification-using-CNN-
       ```
  2.  Create a Virtual Environment
     - It's a best practice to create a virtual environment to keep your project dependencies isolated.
      ```
      # Create the virtual environment
      python -m venv venv
      
      # Activate it
      # On macOS and Linux:
      source venv/bin/activate
      
      # On Windows:
      venv\Scripts\activate
      ```
  3.  Install Dependencies

  4. Download the Dataset
     - Download the Dogs vs. Cats dataset from Kaggle and make sure it is structured in the dataset/ directory as described above.
## Usage
- If you are using a Python script:
  ```
  python train_model.py
  ```
- If you are using a Jupyter Notebook:
  ```
  jupyter notebook Cat_Dog_Classifier.ipynb
  ```
## Result
The model was trained for 25 epochs and achieved a validation accuracy of approximately 85%. The training and validation accuracy/loss curves show that the model learned effectively without significant overfitting.
- Here's an example of the model making a prediction on a new image:
  ```
  # Example prediction code
   import numpy as np
   from tensorflow.keras.preprocessing import image
   
   test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis = 0)
   result = model.predict(test_image)
   
   if result[0][0] == 1:
       prediction = 'dog'
   else:
       prediction = 'cat'
   
   print(f"The image is a {prediction}!")
  ```
