# Flower Image Classifier

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
[![N|Solid](https://github.com/devindatt/Flower_Image_classifier/blob/master/pytorch_flower_classification.png?raw=true)]()



Dillinger is a cloud-enabled, mobile-ready, offline-storage, AngularJS powered HTML5 Markdown editor.

  - Type some Markdown on the left
  - See HTML in the right
  - Magic

# Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python, PyTorch and PIL. The code should run with no issues using Python versions 3.*, PyTorch, PIL.

You can run jupyter notebook to see the entire model build process, or you can run python scripts to directly train and use the model to recognize the flower image.

# Project Motivation

This project is from Udacity's image classification project. It trains CNN model to classify picturesof flowers.

# Project Files

Image Classifier Project.ipynb - It is used to build the model using the jupyter notebook. It can be used independently to see how the model works.
cat_to_name.json - It is used in ipynb and py file to map flower number to flower names.
train.py - It will train a new network on a dataset and save the model as a checkpoint.
predict.py - It uses a trained network to predict the class for an input image.

The project is broken down into multiple steps:

- Load and preprocess the image dataset.
- Train the image classifier on your dataset.
- Use the trained classifier to predict image content.

# Licensing, Authors, Acknowledgements
Credit should be given to Udacity for the project. Usage of this project cannot be used for your Udacity deep learning project. Otherwise, feel free to use the code here that fits your purpose.
