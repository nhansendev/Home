
# Home
A summary of my repositories

### Index

 - Data Science
	 - 3D Printer Filament Customer Review Topic Modeling
	 - Pistachio Image Classification
	 - Applying Painting Style using GAN
	 - Twitter Message Content Classification
	 - Cancerous Cell Detection
	 - Australian Weather Clustering
	 - News Article Topic Classification
	 - Patient Stroke Prediction
  	 - New York Shooting Incidents
 	 - COVID Trends
 - Other
	 - PyDrawNet
 	 - KiCAD autoBOM
   	 - RectanglePack
   	 - PDF Combine
   	 - PyAudioPlayer
   	 - TimeMarkers
   	 - AutoConfig
   	 - DictionaryPrint

# Data Science
## 3D Printer Filament Customer Review Topic Modeling -  [Link](https://github.com/nhansendev/FilamentReviewAnalysis)
![image](https://github.com/user-attachments/assets/04af90b0-702a-4e70-8c98-c8d53aff3530)

**Tools Used:**
 - python
	 - numpy, pandas, matplotlib, sentence_transformers, scikit-learn, nltk, bertopic, scipy, hdbscan, torch, ipywidgets
 - NLP, TF-IDF, Sentence Transformers, Supervised Learning, Clustering Algorithms

**Abstract:**
In this project topic modeling is used to extract actionable insights from product reviews for 3D printer filament. Using this information the factors important to customers when purchasing 3D printer filament can be estimated, as well as more specific feedback on a case-by-case basis, such as per supplier, or filament type. Reviews were retrieved from the AMAZON REVIEWS 2023 dataset after careful filtering was performed to identify relevant products, which required the use of supervised classification algorithms. Topic modeling was performed using the BERTopic model to extract common discussion topics, from which actionable insights could be drawn. Topic comparisons were performed using a variety of metrics, including the frequencies at which topics were paired within reviews, and topic tones. These comparisons revealed several useful insights into customer preferences and common complaints, which could be expanded upon further in future analysis.
	
## Pistachio Image Classification - [Link](https://github.com/nhansendev/DTSA_5511_FinalProject/blob/main/final.ipynb)
![image](https://github.com/user-attachments/assets/aada72bf-b354-4f86-a0ba-075851a2a4d2)
![image](https://github.com/user-attachments/assets/76e7fe2b-ba54-42da-9974-6045eb391182)

**Tools Used:**
 - python
	 - numpy, torch, pillow, scikit-learn, matplotlib, https://github.com/nhansendev/PyDrawNet
 - CNN, Deep Learning, Image Augmentation, Image Classification, CycleGAN (Discriminator)
 
 **Abstract:**
For this project I chose to tackle an image classification problem presented by a Kaggle  [dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset/data). The images in the dataset are of two different varieties of pistachios: "Siirt" and "Kirmizi", with the goal being to create a neural-network based model capable of reliably differentiating between them.

The general steps of the project included exploring and pre-processing the data, preparing the model(s), training the models, and evaluating their performance.

The final model achieved a validation F1-Score of 0.98, indicating that it had effectively learned to classify the pistachios.

## Applying Painting Style using GAN - [Link](https://github.com/nhansendev/DTSA5511_Week5Project/blob/main/project.ipynb)
![image](https://github.com/user-attachments/assets/89e73f2d-96b7-45e4-b8a6-88ed237bbe21)
![image](https://github.com/user-attachments/assets/71e86de2-9d65-4e35-b281-b8a8bdc3d4c0)

**Tools Used:**
 - python
	 - numpy, torch, torchvision, matplotlib, pillow
 - CNN, Deep Learning, Image Augmentation, Image Classification
 
 **Abstract:**
The dataset for this project is provided via the Kaggle "GAN Getting Started"/"I'm Something of a Painter Myself"  [competition](https://www.kaggle.com/competitions/gan-getting-started). The goal of the project is to use a Generative Adversarial Network (GAN) to adapt real photos to the style of Claude Monet, a famous French painter, using examples of his artwork.

To reach this goal the dataset (images) will be explored and pre-processed, then a GAN-based model will be trained and used to generate adapted images. The images will then be submitted for a final score in the competition.

GANs are notoriously difficult to train, and while the model performed well in the numeric Kaggle evaluation, the subjective performance of the model was poor. Despite many manual iterations of hyperparameter tuning, model architecture exploration, and other tweaks, no models capable of "believable" style tranfer arose.

## Twitter Message Content Classification - [Link](https://github.com/nhansendev/DTSA5511_W4/blob/main/project.ipynb)
![image](https://github.com/user-attachments/assets/19244eaf-24e4-44a4-ade9-31e207975e6a)

**Tools Used:**
 - python
	 - numpy, torch, scikit-learn, matplotlib, nltk, spacy, gensim
 - LSTM, NLP, Text Classification
 
 **Abstract:**
The dataset to be analyzed was provided via Kaggle and consists of 10000 Twitter messages hand-classified on whether they are about disasters or not.

The goal of the project was be to clean, explore, and encode the data, then train Recurrent Neural Network (RNN) models to perform Natural Language Processing (NLP) to predict the disaster/not disaster labels.

## Cancerous Cell Detection - [Link](https://github.com/nhansendev/DTSA5511_W3/blob/main/project.ipynb)
![image](https://github.com/user-attachments/assets/11eee528-2b0b-469c-bd4a-c6f995bd5e03)

**Tools Used:**
 - python
	 - numpy, torch, scikit-learn, torchvision, matplotlib
 - CNN, Deep Learning, Image Augmentation, Image Classification
 
 **Abstract:**
For this project the target dataset is a set of images of histopathologic scans (tissue magnified via microscope) of lymph node sections, which may or may not contain metastatic tissue (cancer). The dataset is hosted via Kaggle, with the goal of identifying which images include metatstic tissue.

More specifically, per the data description on Kaggle: "A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image."

Convolutional Neural Network (CNN) based models will be trained towards this goal since they have been proven effective for image-based analysis tasks.

Four models were trained for this goal, with each scoring between 76% - 81% accuracy during testing, with the results likely suffering from over-fitting. Basic ensembling was attempted, but did not improve upon the best individual score.

## Australian Weather Clustering - [Link](https://github.com/nhansendev/DTSA5510_Final/blob/main/project.ipynb)
![image](https://github.com/user-attachments/assets/bb911a0d-f83b-4af8-aedf-6aa16caa746c)

**Tools Used:**
 - python
	 - numpy, pandas, matplotlib, scikit-learn
 - PCA, K-Means, T-SNE
 
 **Abstract:**
For this project I chose a dataset describing weather in Australia, retrieved from Kaggle. The dataset covers about 10 years of daily weather observations from 49 weather stations across Australia. Each observation includes 23 features, such as date, location, temperature, humidity, etc.

Unsupervised clustering analysis will be performed to gain a better understanding of trends in the data.

The KMeans algorithm was used to cluster the weather stations by their weather patterns, resulting in a North-South divide with strong correlations to the maximum daily temperature.

Between the dimensionality reduction algorithms Principle Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (TSNE) it was found that the more complex, non-linear embedding performed by TSNE better captured the structure of the data while translating to lower dimensions. PCA lost most of the available information during the transformation, but still produced plots that could be used to compare relationships between features. Since TSNE is a much slower algorithm, this results in a tradeoff between processing time and embedded feature fidelity.

## News Article Topic Classification - [Link](https://github.com/nhansendev/DTSA5510_Week4_Project/blob/main/week4.ipynb)
![image](https://github.com/user-attachments/assets/405f3190-b6f6-4ed6-a9b6-7c0241981558)
![image](https://github.com/user-attachments/assets/7bc46076-fbc3-4741-af0f-a9a39d47011b)

**Tools Used:**
 - python
	 - numpy, pandas, matplotlib, scikit-learn
 - Regression, Cross-Validation, NMF, PCA
 
 **Abstract:**
The goal of this project was to train unsupervised and supervised models to predict news article topics.

The BBC News Classification dataset used in the analysis was retrieved from Kaggle, and contains 2225 articles in the categories of business, entertainment, politics, sports, and technology.

Logistic Regression models were fit with a testing accuract of ~99%, while NMF models reached ~96%. Observed tradeoffs of the two approaches: 
- Logistic Regression requires labeled data, but is fast to train
- NFM does not require labeled data, but is slower
- Both can achieve similar levels of accuracy for this task

## Patient Stroke Prediction - [Link](https://github.com/nhansendev/DTSA5509_Final/blob/main/final_notebook.ipynb)
![image](https://github.com/user-attachments/assets/71139713-eefa-4e0f-9839-99cd51c8a832)
![image](https://github.com/user-attachments/assets/e4adde2c-c9ae-4645-b571-8fd68ad3f8e5)

**Tools Used:**
 - python
	 - numpy, pandas, matplotlib, scikit-learn
 - Regression, Cross-Validation, KNN, Random Forest, PCA, Descision Tree, SVM
 
 **Abstract:**
For this project I chose to analyze a stroke dataset provided by kaggle. The dataset contains 5110 observations (patients) with 12 attributes, including a binary classification for whether they did or didn't have a stroke. The original source of the data is described as "confidential", and any attributes that might be used to personally identify a patient have been omitted (e.g. name and location). The goal of this project was to develop models capable of accurately predicting which patients are at risk of strokes using the available data. This required cleaning and preprocessing of the data followed by model selection, evaluation, and optimization.

Three methods of data preparation were considered; using PCA to perform dimensionality reduction, using standardized data, and using one-hot encoded data. Models trained on the one-hot encoded data had the highest F1-Scores when predicting on testing data, though there were other models with better recall. Since the consequences of false-negatives could be harmful in a medical setting, it was determined that models with higher false-positive rates were preferable (associated with higher recall). The highest test recall achieved by a model was 0.92, though its precision was 0.18.

Overall, three models were selected for further consideration depending on the requirements of their specific appliction (tradeoffs of precision and recall):
- RandomForest trained on one-hot encoded data
- SVC trained on one-hot encoded data
- An ensemble of the top three models (RandomForest, AdaBoost, and LogisticRegression, each trained on one-hot encoded data)

## New York Shooting Incidents - [Link](https://html-preview.github.io/?url=https://github.com/nhansendev/DTSA5301_Final/blob/main/NY_Data.html)
![image](https://github.com/user-attachments/assets/9c5bc25c-7237-4ef7-b3d2-b4f0fa8dc754)

**Tools Used:**
 - R
 
 **Abstract:**
 The goal of this project was to identify trends in shooting incident data (retrieved from the city of New York [website](https://opendata.cityofnewyork.us/)), which required data to be imported, cleaned, and analyzed.

 ## COVID Trends - [Link](https://html-preview.github.io/?url=https://github.com/nhansendev/DTSA5301_Final/blob/main/COVID_Data.html)
![image](https://github.com/user-attachments/assets/32854f18-b6dd-4ec7-9d8d-5a8c0ab7ed06)

**Tools Used:**
 - R
 
 **Abstract:**
 The goal of this project was to identify trends in COVID-19 data (retrieved from John Hopkins University), which required data to be imported, cleaned, and analyzed.	


# Other

## PyDrawNet - [Link](https://github.com/nhansendev/PyDrawNet)
A python utility for plotting neural network (and other) diagrams
![image](https://github.com/user-attachments/assets/42dabde1-b3ab-4474-a352-0f524983f527)

## KiCAD autoBOM - [Link](https://github.com/nhansendev/KiCAD_autoBOM)
Python scripts for automating BOM operations in KiCAD
![image](https://github.com/user-attachments/assets/3bdc5348-25a5-457d-ab29-100b96e0bc28)

## RectanglePack - [Link](https://github.com/nhansendev/RectanglePack)
This Python project expands on the capabilities of the rectangle-packer package primarily by adding efficient rotation checking (missing from the base package), the ability to maximize area usage of stock, and multi-sheet packing.

![image](https://github.com/user-attachments/assets/b3c66ee1-f97a-42d1-90c3-46171b375029)

## PDF Combine - [Link](https://github.com/nhansendev/PDFCombine)
A python utility for automatically combining and trimming PDF files
![image](https://github.com/user-attachments/assets/60f755b0-abc7-496f-b8a7-3405410b38c3)

## PyAudioPlayer - [Link](https://github.com/nhansendev/PyAudioPlayer)
An audio player GUI with yt_dlp integration, made in python using PySide6.
![image](https://github.com/user-attachments/assets/183034ac-0dae-40ab-a3e0-f8a472b8d6b9)

## TimeMarkers - [Link](https://github.com/nhansendev/TimeMarkers)
Some simple Python time tracking utilities

## AutoConfig - [Link](https://github.com/nhansendev/AutoConfig)
A python utility for reading/writing custom YAML configuration files

## DictionaryPrint - [Link](https://github.com/nhansendev/DictionaryPrint)
A python utility for printing out dictionary contents in an easily readable format.
