import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import string
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import face_recognition
import pickle
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import storage
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from flask import Flask, jsonify, Response, render_template


from decouple import config
DATABASE_URL = config('DATABASE_URL')
STORAGE_BUCKET = config('STORAGE_BUCKET')
def perform_auto_grading():
    cred = credentials.Certificate('credentials.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL,
        'storageBucket': STORAGE_BUCKET
    })

    # Get a Realtime Database client
    ref = db.reference('Students')
    X = [
        "Friendship is one of the greatest bonds anyone can ever wish for. Lucky are those who have friends they can trust. Friendship is a devoted relationship between two individuals. They both feel immense care and love for each other.",
        "A true friendship makes life easy and gives us good times. Thus, when the going gets tough, we depend on our friends for solace. Sometimes, it is not possible to share everything with family, that is where friends come in.",
        "Irrespective of all differences, a friend chooses you, understands you, and supports you. Whenever you are in self-doubt or lacking confidence, talk to a friend, and your worry will surely go away."
    ]
    y = [12, 14, 13]  # Load the corresponding grades

    # Define the grading criteria
    features = ['spelling', 'grammar', 'sentence_structure', 'coherence']

    def extract_vocabulary_richness(text):
        words = text.split()
        unique_words = set(words)
        return len(unique_words) / len(words)

    def extract_sentence_variety(text):
        sentences = text.split('. ')
        unique_sentences = set(sentences)
        return len(unique_sentences) / len(sentences)

    # Train multiple classifiers on the dataset
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X)

    vocabulary_richness = np.array([extract_vocabulary_richness(x) for x in X])
    sentence_variety = np.array([extract_sentence_variety(x) for x in X])
    X_train = np.column_stack((X_train.toarray(), vocabulary_richness, sentence_variety))

    clf1 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001, solver='adam',
                        verbose=10, random_state=21, tol=0.0001, learning_rate_init=0.001)
    clf2 = MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=500, alpha=0.0001, solver='adam',
                        verbose=10, random_state=21, tol=0.0001, learning_rate_init=0.001)
    clf3 = MLPClassifier(hidden_layer_sizes=(150,), max_iter=500, alpha=0.0001, solver='adam',
                        verbose=10, random_state=21, tol=0.0001, learning_rate_init=0.001)

    # Use VotingClassifier to combine the models
    voting_clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting='hard')
    voting_clf.fit(X_train, y)

    # Process multiple images and extract the text
    image_filenames = ['img/852741.png', 'img/458392.png', 'img/963852.jpg']
    for image_filename in image_filenames:
        # Load the image and apply grayscale and resize operations
        img = cv2.imread(image_filename)
        # Extract text from the image
        text = pytesseract.image_to_string(img)

        def preprocess_text(text):
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Convert to lowercase
            text = text.lower()
            return text

        # Preprocess the text
        processed_text = preprocess_text(text)
        vocabulary_richness = extract_vocabulary_richness(processed_text)
        sentence_variety = extract_sentence_variety(processed_text)
        X_test = vectorizer.transform([processed_text])
        X_test = np.column_stack((X_test.toarray(), vocabulary_richness, sentence_variety))

        # Use the trained model to predict the grade
        predicted_grade = voting_clf.predict(X_test)[0]
        image_id = image_filename.split('/')[-1].split('.')[0]
        print(f"The predicted grade for {image_id} is: {predicted_grade}")
        # Save the grade to the 'cie' field of the student record in the 'Students' node
        student_id = os.path.basename(image_id)
        ref.child(student_id).update({
            'cie': int(predicted_grade)
        })