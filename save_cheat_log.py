import cv2
from PIL import Image
import csv
import os
import pandas as pd

IMG_PATH = "./cheat_logs/images/"
os.makedirs(IMG_PATH, exist_ok=True)

SIZE = (128, 128)

def save_image_log(image, timestamp, cheatingtype):
    # Generate a filename using timestamp and cheating type
    t = timestamp.strftime("%d_%m_%Y_%H__%M_%S")
    filename = f"{t}-{cheatingtype}.jpg"
    location = IMG_PATH + filename

    # Resize and convert the image 
    resized = cv2.resize(image, SIZE)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im_pil.save(location)

    log_time = timestamp.strftime("%d/%m/%Y %H:%M:%S")
    log = [log_time, cheatingtype, location]

    # Ensure log_files directory exists
    log_dir = "./cheat_logs/log_files/"
    os.makedirs(log_dir, exist_ok=True)  

    with open(log_dir + "cheat_records.txt", "a") as logfile:
        logfile.write(log)

def action_recording(result):
    with open('behavior_tracking.txt', 'w') as f:
        f.write("index\tscore\taction\ttab\n")
        for i in range(len(result.face_blendshapes[0])):
            index = result.face_blendshapes[0][i].index
            score = result.face_blendshapes[0][i].score
            action = result.face_blendshapes[0][i].category_name
            tab = None
            f.write(f"{index}\t{score}\t{action}\t{tab}\n")
