My Data Science & AI Portfolio
Welcome! This repository documents my journey and hands-on projects in the fields of Machine Learning, Deep Learning, and Computer Vision. It showcases a progression from foundational data analysis with classical algorithms to building and implementing state-of-the-art neural networks for complex tasks like real-time object detection.

🚀 Table of Contents
Classical Machine Learning

Computer Vision

Deep Learning for Image Classification

Haar Cascade Object Detection

YOLO Object Detection

Skills Demonstrated

1. Classical Machine Learning
This section covers a foundational machine learning project focused on data analysis, feature engineering, and predictive modeling using traditional algorithms.

Project: Titanic Survival Prediction
File: Titanic_Survival_Prediction.ipynb

Description: A comprehensive analysis of the classic Titanic dataset. This project involves a full data science workflow, including:

Exploratory Data Analysis (EDA) to understand passenger demographics and their relationship with survival rates.

Data Cleaning and Preprocessing to handle missing values (NaNs) and convert categorical features into a machine-readable format.

Feature Engineering to create insightful new variables from existing data.

Building and evaluating a classification model using scikit-learn to predict passenger survival.

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

2. Computer Vision
This section is dedicated to my work in computer vision, exploring techniques from classic algorithms to modern, deep learning-based models.

2.1. Deep Learning for Image Classification
This area focuses on using neural networks to classify images, starting with building a Convolutional Neural Network (CNN) from scratch and then leveraging powerful pre-trained models.


Licensed by Google
Intro to Deep Learning (Fashion MNIST)

File: Intro_to_Deep_Learning_Fashion_MNIST.ipynb

Description: My first step into deep learning. This notebook builds a basic sequential neural network using TensorFlow/Keras to classify 10 different types of clothing from the Fashion MNIST dataset.

Building a CNN from Scratch (CIFAR-10)

File: CNN_for_Image_Classification_CIFAR10.ipynb

Description: A hands-on project to construct a Convolutional Neural Network (CNN) from the ground up. The model is trained to classify images from the diverse CIFAR-10 dataset into 10 categories (e.g., airplane, car, bird).

Transfer Learning with MobileNet & VGG16

Files: Transfer_Learning_with_MobileNet.ipynb, Transfer_Learning_with_VGG16.ipynb

Description: These notebooks demonstrate the power and efficiency of transfer learning. I use industry-standard, pre-trained models (MobileNet and VGG16) as a feature-extraction base and fine-tune them for new image classification tasks. This is a key technique for achieving high accuracy with limited data.

2.2. Haar Cascade Object Detection
A classic, lightweight approach to object detection that uses feature-based cascades. It's highly effective for detecting objects with well-defined structural features.

Face and Eye Detection

File: Face_and_Eye_Detection.ipynb

Description: Implements pre-trained Haar Cascade classifiers from OpenCV to first detect human faces in an image and then identify the eyes within each detected face.

License Plate Detection

File: License_Plate_Detection.ipynb

Description: Utilizes a specialized Haar Cascade model trained to detect Russian license plates in images of vehicles.

2.3. YOLO (You Only Look Once) Object Detection
YOLO is a state-of-the-art object detection system known for its exceptional speed and accuracy, making it ideal for real-time applications.

YOLOv3 for Image and Video Analysis

Files: YOLOv3_Image_Detection.ipynb, YOLOv3_Video_Detection.ipynb

Description: These projects use the YOLOv3-tiny model with OpenCV's DNN module to perform object detection on both static images and video files. The model can identify and draw bounding boxes around 80 different object classes.

YOLOv8 for Real-Time Tracking

File: YOLOv8_Live_Tracking.ipynb

Description: This notebook leverages the modern and powerful ultralytics library to implement real-time object detection and tracking using the YOLOv8n model. It showcases high-performance inference suitable for live video streams.

3. Skills Demonstrated
Data Science & ML: Data Cleaning, EDA, Feature Engineering, Predictive Modeling, Pandas, Scikit-learn.

Deep Learning: Neural Networks, Convolutional Neural Networks (CNNs), Transfer Learning, TensorFlow, Keras.

Computer Vision: Image Classification, Object Detection, Real-Time Video Analysis, OpenCV.

Model Implementation: Haar Cascades, YOLOv3, YOLOv8, MobileNet, VGG16.

Tools: Python, Jupyter Notebook, Git & GitHub.