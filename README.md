# Depression-Detection-Using-NLP-Web-App
Depression Detection Using NLP
This project is a part of final year project done by the Computer Science students of Sahyadri College of Engineering and Management, Mangaluru.
-- Project Status: [Completed]

Project Intro/Objective
The purpose of this project is to detect whether a person is depressed or not by analyzing the post which he posts on social media in the form of text or images.
Partner
• Rathan H V
• Raviganesh M
• Shyam Kishore
Methods Used
• Machine Learning
• Data Visualization
• Predictive Modeling
Technologies
• Python
• Pandas, Jupyter, Spyder
• HTML
• JavaScript
• Firebase Database


Project Description
The data collection is done from the Kaggle website. The “Sentiment Analysis Dataset” is extracted from the posts of Twitter social media platform. The data is pre-processed and fed into a Sequential ML model.
A front-end for the user is provided as a website. The user is allowed to input the posts in the form of text, voice or image. The user input is fed into the ML model and it will predict whether the depression is present in the user inputted text or not. The result will be displayed on a new page of the website.
Getting Started
To get started, you'll want to first clone the GitHub repository locally or if you have project file, that’s fine.

$ git clone https://github.com/RAVIGANESHMBHAT/Depression-Detection-Using-NLP-Web-App.git

Next, you'll want to go into the sample app directory:

$ cd Depression-Detection-Using-NLP-Web-App/Depression Detection Web App

Then you'll want to install all of the Python requirements (via pip):

$ pip install -r requirements.txt

Note: Search online for the installation of pytesseract OCR which requires some additional steps. ( You can refer this video  https://www.youtube.com/watch?v=RewxjHw8310  )

Download the tokenizer.pkl, w2v_model2.pkl, model_depression.h5 files from the below given link to google drive and move all the three files to the “Depression Detection Web App” folder.
(link to google drive) 

And lastly, you'll want to run the app.py script which will guide you the rest of the way:

$ python app.py
