# Vjezba 3 - prepoznavanje lica

# print(cv.__file__)
# print(cv.data.haarcascades)

import cv2 as cv

# original_image = cv.imread('moje_lice.jpg')

# grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# detected_faces = face_cascade.detectMultiScale(grayscale_image)

# for (column, row, width, height) in detected_faces:
#     cv.rectangle(
#         original_image,
#         (column, row),
#         (column + width, row + height),
#         (0, 255, 0),
#         2
#     )

# cv.imshow('Image', original_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()
