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

import autograde
# import app
# Initialize Firebase Admin SDK with credentials
app = Flask(__name__)
autograde.perform_auto_grading()

bucket = storage.bucket()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# # print(len(imgModeList))

print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
print("loaded")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []
# studentInfo = {}  # Initialize studentInfo variable
def generate_frames():
    global cap, imgBackground, modeType, counter, id, imgStudent, studentInfo

    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        # Resize img to match the shape of the target region in imgBackground
        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)
                # print("Match Index", matchIndex)
                if matches[matchIndex]:
                    # print("known face")
                    # print(studentIds[matchIndex])
                    y1, x2, y2, x1 = faceLoc
                    face_height = y2 - y1
                    face_width = x2 - x1
                    min_face_size = 50  # adjust this to your desired threshold
                    if face_height < min_face_size or face_width < min_face_size:
                        continue
                    else:
                        # y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=2)
                        id = studentIds[matchIndex]
                        if counter == 0:
                            counter = 1
                            modeType = 1
            if counter != 0:

                if counter == 1:
                    studentInfo = db.reference(f'Students/{id}').get()
                    print(studentInfo)

                    # getting image from storage
                    blob = bucket.get_blob(f'images/{id}.png')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                    # updating attendance

                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    print(secondsElapsed)
                    if secondsElapsed > 20:
                        ref = db.reference(f'Students/{id}')
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        # counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:
                    if 10 < counter < 20:
                        modeType = 2
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['cie']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = {}
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        else:
            modeType = 0
            counter = 0
        # cv2.imshow('Webcam', img)
        # cv2.imshow('Face attendance', imgBackground)
        # key = cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', imgBackground)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# @app.route('/ip')
# def video_feed():
#     bucket = storage.bucket()
#     blob = bucket.blob('images/your_image.png')
#     image_url = blob.generate_signed_url(expiration=timedelta(minutes=5), version="v4")
#
#     return image_url

# @app.route('/hello')
# def hello_world():
#     cred = credentials.Certificate("credentials.json")
#     bucket = storage.bucket()
#     return "bakaana"
if __name__ == '__main__':
    app.run()






# modeType = 0
# counter = 0
# id = -1
# imgStudent = []
# while True:
#     success, img = cap.read()
#
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
#     # Resize img to match the shape of the target region in imgBackground
#     imgBackground[162:162 + 480, 55:55 + 640] = img
#     imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
#
#     if faceCurFrame:
#         for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#
#             matchIndex = np.argmin(faceDis)
#             # print("Match Index", matchIndex)
#             if matches[matchIndex]:
#                 # print("known face")
#                 # print(studentIds[matchIndex])
#                 y1, x2, y2, x1 = faceLoc
#                 face_height = y2 - y1
#                 face_width = x2 - x1
#                 min_face_size = 50  # adjust this to your desired threshold
#                 if face_height < min_face_size or face_width < min_face_size:
#                     continue
#                 else:
#                     # y1, x2, y2, x1 = faceLoc
#                     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#                     bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
#                     imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=2)
#                     id= studentIds[matchIndex]
#                     if counter==0:
#                         counter=1
#                         modeType=1
#         if counter!=0:
#
#             if counter==1:
#                 studentInfo = db.reference(f'Students/{id}').get()
#                 print(studentInfo)
#
#                 #getting image from storage
#                 blob = bucket.get_blob(f'images/{id}.png')
#                 array = np.frombuffer(blob.download_as_string(), np.uint8)
#                 imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
#
#                 #updating attendance
#
#                 datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
#                                                    "%Y-%m-%d %H:%M:%S")
#                 secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
#                 print(secondsElapsed)
#                 if secondsElapsed>20:
#                     ref = db.reference(f'Students/{id}')
#                     studentInfo['total_attendance'] += 1
#                     ref.child('total_attendance').set(studentInfo['total_attendance'])
#                     ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#                 else:
#                    modeType = 3
#                    # counter = 0
#                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
#
#             if modeType!=3:
#                 if 10<counter<20:
#                     modeType=2
#                 imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
#
#                 if counter<=10:
#                     cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#                     cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(imgBackground, str(id), (1006, 493),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
#                     cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
#                     cv2.putText(imgBackground, str(studentInfo['year']), (1125, 625),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
#
#                     (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
#                     offset = (414 - w) // 2
#                     cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
#                     imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
#             counter+=1
#
#             if counter>=20:
#                 counter = 0
#                 modeType = 0
#                 studentInfo = {}
#                 imgStudent = []
#                 imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
#     else:
#         modeType = 0
#         counter = 0
#     # cv2.imshow('Webcam', img)
#     cv2.imshow('Face attendance', imgBackground)
#     key = cv2.waitKey(1)
# # Check if the user closed the window
#     if key == 27 or cv2.getWindowProperty('Face attendance', cv2.WND_PROP_VISIBLE) == 0:
#         break
# #Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()
#







#code for converting handwritten sentences in an image to text
# import cv2
# import pytesseract
#
# # Load the image using OpenCV
# img = cv2.imread('img/ess.jpg')
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply adaptive thresholding to enhance the text visibility
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
# # Apply dilation to fill gaps in the text
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# dilation = cv2.dilate(thresh, kernel, iterations=1)
#
# # Use pytesseract to convert the image to text with the LSTM-based OCR model
# text = pytesseract.image_to_string(dilation, config='--psm 6 --oem 1')
#
# # Print the extracted text
# print(text)
#
