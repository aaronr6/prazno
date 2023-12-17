import cv2 as cv
import matplotlib.pyplot as plt 
from round_rectangle import *
from new_line import *
from deepface import DeepFace

img = cv.imread('aaron_2.jpg')

img_temp = img.copy()
unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
dominant_color = tuple(unique[np.argmax(counts)])

analysis = DeepFace.analyze(img, actions = ['age', 'gender', 'race', 'emotion'])
analysis = analysis[0]
# print(analysis)
# print(analysis["age"], "years old", analysis["dominant_race"], " ", analysis["dominant_emotion"], " ", analysis["dominant_gender"])

parameters = analysis["region"]
# print(parameters)

draw_border(img, (parameters["x"],parameters["y"]), (parameters["x"]+parameters["w"], parameters["y"]+parameters["h"]), (226, 204, 120), 5, 10, 20)

text = f'''AGE: {analysis["age"]}
RACE: {analysis["dominant_race"]}
EMOTION: {analysis["dominant_emotion"]}
GENDER: {analysis["dominant_gender"]}'''

font = cv.FONT_HERSHEY_SIMPLEX 
position = (parameters["x"]+parameters["w"]+10, parameters["y"]) 
font_scale = 1
font_color = (102, 80, 0)
line_thickness = 2

text_size, baseline = cv.getTextSize(text, font, font_scale, line_thickness)
text_width = text_size[0]
text_width = int(text_width)

height, width, channels = img.shape
width = int(width)

print(type(dominant_color))

if width + text_width + 10 > width:
    img = cv.copyMakeBorder(img, 0, 0, 0, width, cv.BORDER_CONSTANT, value = dominant_color)

add_text_to_image(img, text, position, font_scale, line_thickness, font, font_color)

cv.imwrite('detected.png', img)