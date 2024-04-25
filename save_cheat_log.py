import cv2
from PIL import Image
import csv
import os

PATH = "./cheat_logs/images/"
os.makedirs(PATH, exist_ok=True)

SIZE = (128, 128)

def save_image_log(image, timestamp, cheatingtype):
    # Generate a filename using timestamp and cheating type
    t = timestamp.strftime("%d_%m_%Y_%H__%M_%S")
    filename = f"{t}-{cheatingtype}.jpg"
    location = PATH + filename

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

    with open(log_dir + "cheat_records.csv", "a") as logfile:
        writer_obj = csv.writer(logfile)
        writer_obj.writerow(log)
