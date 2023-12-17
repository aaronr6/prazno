import cv2 as cv
from deepface import DeepFace 
from round_rectangle import *
from new_line import *
from deepface import DeepFace

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:

        face = frame[y:y + h, x:x + w]

        result = DeepFace.analyze(face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
        result = result[0]

        text_age = f"AGE: {result['age']}"
        text_gender = f"GENDER: {result['dominant_gender']}"
        text_emotion = f"EMOTION: {result['dominant_emotion']}"
        text_race = f"RACE: {result['dominant_race']}"

        draw_border(frame, (x, y), (x + w, y + h), (226, 204, 120), 5, 10, 20)

        cv.putText(frame, text_age, (x + w + 10, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.putText(frame, text_gender, (x + w + 10, y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.putText(frame, text_emotion, (x + w + 10, y + 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.putText(frame, text_race, (x + w + 10, y + 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow('Face Recognition', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
