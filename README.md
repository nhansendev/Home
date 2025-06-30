
# Home
A summary of my repositories

### Index

 - Data Science
	 - [Filament Review Analysis](https://github.com/nhansendev/Home/tree/main?tab=readme-ov-file#filament-review-analysis----link)
	 - [Deep Learning Project - Pistachio Image Classification](https://github.com/nhansendev/Home/tree/main?tab=readme-ov-file#deep-learning-project---pistachio-image-classification---link)
	 - [Applying Painting Style using GAN](https://github.com/nhansendev/Home/blob/main/README.md#applying-painting-style-using-gan---link)
	 - [PyDrawNet](https://github.com/nhansendev/Home/tree/main?tab=readme-ov-file#pydrawnet---link)
 - Other
 - 

# Data Science
## Filament Review Analysis -  [Link](https://github.com/nhansendev/FilamentReviewAnalysis)
![image](https://github.com/user-attachments/assets/04af90b0-702a-4e70-8c98-c8d53aff3530)

**Tools Used:**
 - python
	 - numpy, pandas, matplotlib, sentence_transformers, scikit-learn, nltk, bertopic, scipy, hdbscan, torch, ipywidgets
 - NLP, TF-IDF, Sentence Transformers, Supervised Learning, Clustering Algorithms

**Abstract:**
In this project topic modeling is used to extract actionable insights from product reviews for 3D printer filament. Using this information the factors important to customers when purchasing 3D printer filament can be estimated, as well as more specific feedback on a case-by-case basis, such as per supplier, or filament type. Reviews were retrieved from the AMAZON REVIEWS 2023 dataset after careful filtering was performed to identify relevant products, which required the use of supervised classification algorithms. Topic modeling was performed using the BERTopic model to extract common discussion topics, from which actionable insights could be drawn. Topic comparisons were performed using a variety of metrics, including the frequencies at which topics were paired within reviews, and topic tones. These comparisons revealed several useful insights into customer preferences and common complaints, which could be expanded upon further in future analysis.
	
## Deep Learning Project - Pistachio Image Classification - [Link](https://github.com/nhansendev/DTSA_5511_FinalProject/blob/main/final.ipynb)
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


## PyDrawNet - [Link](https://github.com/nhansendev/PyDrawNet)
A python utility for plotting neural network (and other) diagrams
![image](https://github.com/user-attachments/assets/42dabde1-b3ab-4474-a352-0f524983f527)

# Other


> Written with [StackEdit](https://stackedit.io/).
