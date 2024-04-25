import cv2
from time import sleep
import numpy as np

from src.screen.models.yolos import ObjectDetector

cap = cv2.VideoCapture(0)

# YOLOS is a Vision Transformer (ViT) trained using the DETR loss
detector = ObjectDetector(model_path='hustvl/yolos-base')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    else:
        outputs = detector.detect(frame, threshold=0.5)

        person = 0
        for i in range(outputs[0]):
            '''
            We only need to assert if there exists another person or cell_phone being used
            '''
            if outputs[0][i] == 'person':
                person += 1
            if outputs[0][i] == 'cell_phone':
                print('Phone detected')
            if outputs[0][i] == 'mask':
                print('Please remove mask')
        if person == 0:
            print('No candidate detected, please turn back !!!')

        detector.draw_outputs(frame, outputs)

        cv2.imshow('Face_detecting', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
