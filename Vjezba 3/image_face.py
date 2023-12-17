import cv2 as cv
import matplotlib.pyplot as plt 
from round_rectangle import *
from new_line import *
from deepface import DeepFace

def run_image():

    img = cv.imread('aaron_2.jpg')

    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    dominant_color = list(unique[np.argmax(counts)])
    print(dominant_color)
    dominant_color2 = []
    for i in dominant_color:
        dominant_color2.append(int(i))


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

    check_text1 = f"EMOTION: {analysis['dominant_emotion']}"
    check_text2 = f"RACE: {analysis['dominant_race']}"

    font = cv.FONT_HERSHEY_SIMPLEX 
    position = (parameters["x"]+parameters["w"]+10, parameters["y"]) 
    font_scale = 1
    font_color = (102, 80, 0)
    line_thickness = 2

    text_size1, baseline1 = cv.getTextSize(check_text1, font, font_scale, line_thickness)
    text_width1 = text_size1[0]
    text_width1 = int(text_width1)

    text_size2, baseline2 = cv.getTextSize(check_text2, font, font_scale, line_thickness)
    text_width2 = text_size2[0]
    text_width2 = int(text_width2)

    height, width, channels = img.shape
    width = int(width)

    if parameters["x"]+parameters["w"] + text_width1 + 10 > width or parameters["x"]+parameters["w"] + text_width2 + 10 > width:
        img = cv.copyMakeBorder(img, 0, 0, 0, width, cv.BORDER_CONSTANT, value = dominant_color2)

    add_text_to_image(img, text, position, font_scale, line_thickness, font, font_color)

    cv.imwrite('detected.png', img)