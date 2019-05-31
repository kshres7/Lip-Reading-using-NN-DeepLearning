Lip Reading using Neural Network and Deep Learning 

By Karan Shrestha

Undergraduate - CS senior capstone project  

This project provides an implementation of deep learning neural network capable of training a lip reading model. The trained model is used to predict words from the speaker with input image frames using web camera in a web application. 

Dataset: The Oxford-BBC Lip Reading in the Wild (LRW) Dataset 

Source Link: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

This repository contains two main directories - Lip_Reading_Master and web_application. Lip_Reading_Master includes the implementation of the deep learning neural network used to train the model, preprocessed data (in NPY directory) and XML files for Haar Cascade Classifier. 

web_application directory contains the source code for web application - HTML, CSS and JavaScript files, also providing the pre-trained deep learning model.

Dependencies 

Keras 2.0.9
Tensorflow  1.11.0
Python 3.0+
OpenCV 4.0
Scikit-learn 0.20+
PIP (install packages)

Data Preprocessing 

To start the preprocessing the data run data_preprocessing.py file. Change the input path of the data set using video_path = ‘path_to_video_source’

To compile and run the code, open terminal in the project directory and run the command

 $ python data_preprocessing.py

To check the saved samples of the preprocessed data. Run this after the preprocessing is complete.  

 $ python visualization_preprocess.py 

Training

Initially the Model D architecture is set to train on the preprocessed data. The model can also be trained with other provided architectures.

$ python model_training.py

Evaluation

Check for the results to evaluate the trained model. Once, the training is completed, simply run the command: 

$ python model_evaluation.py

Web Application 

Run the web application in a local HTTP server - open terminal window and navigate to the web_application directory. 

$ python -m http.server 8000

Then, open a web browser (preferred - Google Chrome and Mozilla Firefox) and type localhost:8000  


Note: The model was trained using Keras version 2.2.4 which no longer supports to apply the model weights in keras.js (version 0.3.0). A new model needs to be trained using Keras version 2.0.9 to attach the model weights to show results in the word prediction bar plot in the provided web application. 