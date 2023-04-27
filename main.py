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

# Initialize Firebase Admin SDK with credentials
cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://pommy-a7657-default-rtdb.firebaseio.com/",
    'storageBucket': "pommy-a7657.appspot.com"
})

# Get a Realtime Database client
ref = db.reference('Students')

X = [
    "Friendship is one of the greatest bonds anyone can ever wish for. Lucky are those who have friends they can trust. Friendship is a devoted relationship between two individuals. They both feel immense care and love for each other.",
    "A true friendship makes life easy and gives us good times. Thus, when the going gets tough, we depend on our friends for solace. Sometimes, it is not possible to share everything with family, that is where friends come in.",
    "Irrespective of all differences, a friend chooses you, understands you, and supports you. Whenever you are in self-doubt or lacking confidence, talk to a friend, and your worry will surely go away."
]
y = [6, 7, 7]  # Load the corresponding grades

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

cred = credentials.Certificate("credentials.json")
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
# print(len(imgModeList))


print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

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
                    id= studentIds[matchIndex]
                    if counter==0:
                        counter=1
                        modeType=1
        if counter!=0:

            if counter==1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                #getting image from storage
                blob = bucket.get_blob(f'images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                #updating attendance

                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed>30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                   modeType = 3
                   # counter = 0
                   imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType!=3:
                if 10<counter<20:
                    modeType=2
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter<=10:
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
            counter+=1

            if counter>=20:
                counter = 0
                modeType = 0
                studentInfo = []
                imgStudent = []
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0
    # cv2.imshow('Webcam', img)
    cv2.imshow('Face attendance', imgBackground)
    key = cv2.waitKey(1)
    # Check if the user closed the window
    if key == 27 or cv2.getWindowProperty('Face attendance', cv2.WND_PROP_VISIBLE) == 0:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()


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
