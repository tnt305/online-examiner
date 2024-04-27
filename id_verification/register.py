import cv2
import numpy as np
import os
from datetime import datetime


def add_overlay_and_holder(frame):
    # Add a 80% black transparent overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay.fill(0)
    alpha = 0.8
    frame_height, frame_width = frame.shape[:2]
    
    # Add a holder
    holder_height, holder_width = 240, 320
    holder = np.zeros_like(frame, dtype=np.uint8)
    holder.fill(255)  # White color
    holder_x = (frame_width - holder_width) // 2
    holder_y = (frame_height - holder_height) // 2
    holder_thickness = 5  # Thickness of white border around the visual
    holder[holder_y:holder_y+holder_height, holder_x:holder_x+holder_width] = 0  # Black out the center
    holder[holder_y+holder_thickness:holder_y+holder_height-holder_thickness,
           holder_x+holder_thickness:holder_x+holder_width-holder_thickness] = 255  # Leave a white border
    
    # Apply the overlay to the frame
    overlayed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Delete the overlay in the center area
    overlayed_frame[holder_y:holder_y+holder_height, holder_x:holder_x+holder_width] = frame[holder_y:holder_y+holder_height, holder_x:holder_x+holder_width]
    
    return overlayed_frame

def main():
    cap = cv2.VideoCapture(0)  # Change 0 to the camera index if you have multiple cameras
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add overlay and holder to the frame
        frame_with_overlay = add_overlay_and_holder(frame)
        
        # Show the resulting frame
        cv2.imshow('Camera with Overlay and Holder', frame_with_overlay)
        
        # Save the image area 320x240 if 'r' is pressed
        key = cv2.waitKey(1)
        if key == ord('r'):
            # Get the region of interest (ROI)
            frame_roi = frame[60:300, 160:480]  # 320x240 starting from (160, 60)
            
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            path = 'data/login_check/'
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(f'{path}/save_image_{timestamp}.jpg', frame_roi)
        
        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
