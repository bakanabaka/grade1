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

def load_encode():
    print("Loading Encode File ...")
    file = open('EncodeFile.p', 'rb')
    print("loaded")
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    print(studentIds)
    print("Encode File Loaded")
