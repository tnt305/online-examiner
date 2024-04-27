import numpy as np
import cv2

def enhance_details_and_sharpen(image, detail_strength=1.0, sharpen_strength=1.0, blur_radius=1.0):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray_image, (2, 2), blur_radius)
    
    # Calculate the difference between the original grayscale image and the blurred image
    detail = gray_image - blurred
    
    # Enhance the details by adding the detail back to the original image
    enhanced = cv2.addWeighted(gray_image, 1.0 + detail_strength, detail, -detail_strength, 0)
    
    # Apply sharpening to the enhanced image
    sharpened = cv2.filter2D(enhanced, -1, np.array([[-1, -1, -1],
                                                     [-1,  9, -1],
                                                     [-1, -1, -1]]))
    
    # Convert the sharpened image back to BGR (if input image was color)
    if len(image.shape) == 3:
        sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return sharpened
