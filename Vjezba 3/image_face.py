import cv2 as cv
import matplotlib.pyplot as plt 
from round_rectangle import *
from new_line import *
from deepface import DeepFace

img = cv.imread('aaron_2.jpg')

plt.imshow(img[:, :, : : -1])

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

add_text_to_image(img, text, position, font_scale, line_thickness, font, font_color)

cv.imwrite('detected.png', img)