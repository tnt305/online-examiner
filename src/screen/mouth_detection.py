import mediapipe as mp
import time
import cv2
import numpy as np
#import pandas as pd

'''
Originally from 
https://github.com/FedeClaudi/LookMaNoHands/blob/82ceff9e6b346f4e6720484f7e0877bf66f07020/archive/live_tracking.py#L24
and 
https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python#live-stream
'''

from src.screen.models.face_landmark import * 

# define a global variable to store the results
results = None

# Create a face landmarker instance with the live stream mode:
def store(new_result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global results
    results = new_result
    

options = FaceLandmarkerOptions(
    base_options= BaseOptions(model_asset_path= "face_landmarker_v2_with_blendshapes.task"),
    running_mode= VisionRunningMode.LIVE_STREAM,
    result_callback=store,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)



with FaceLandmarker.create_from_options(options) as landmarker:
    # Set up video capture from default camera
    cap = cv2.VideoCapture(0)

    # set video fps to 60
    cap.set(cv2.CAP_PROP_FPS, 60)
    print("Starting")
    # Set up FPS counter
    frames = 0
    start_time = time.time()

    #   The landmarker is initialized. Use it here.
    while cap.isOpened():
        ret, frame = cap.read()
        # Display frame
        frames += 1

        # Calculate FPS
        if frames > 10:
            fps = round(frames / (time.time() - start_time), 2)
        else:
            fps = 0
        frame = cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Send live image data to perform face landmarking.
        # The results are accessible via the `result_callback` provided in
        # the `FaceLandmarkerOptions` object.
        # The face landmarker must be created with the live stream mode.\
        frame_timestamp_ms = int(round(time.time() * 1000))
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if results is not None:
            if len(results.face_landmarks) == 0:
                print("No face detected, please back to seat")
                
            frame = draw_landmarks_on_image(frame, results)


            # check if a recognized action is being made
            if len(results.face_blendshapes) >1 :
                print(f'{len(results.face_blendshapes)} people detected')

            suspicious = 0
            eye_behaviors = []
            for item in results.face_blendshapes[0]:
                if item.category_name.startswith('mouth') and 0.05 <= item.score < 0.4:
                    suspicious += 1

                if item.category_name.startswith(('eyeLookDown', 'eyeLookUp')):
                    if item.score > max_score:
                        max_score = item.score
                        index = item.index
                        eye_behaviors.append(item.category_name)

            while len(eye_behaviors) >= 100:
                eye_behaviors = eye_behaviors[:-100]
                eye_behaviors_series = list(eye_behaviors)
                # Custom count logic
                eye_behavior_counts = {behavior: eye_behaviors_series.count(behavior) for behavior in set(eye_behaviors_series)}

                # Check the difference between counts of the two values
                if abs(eye_behavior_counts[0] - eye_behavior_counts[1]) <= 20:
                    suspicious += 1
                break

            if suspicious >= 10:
                print('Cheating based cheat sheet suspected')

                    

        # show frame
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break