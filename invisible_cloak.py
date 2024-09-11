# Filename: invisible_cloak.py

import numpy as np
import cv2
import time

def capture_background(cap, num_frames=50):
    """
    Captures the background frame to use for creating the invisibility effect.

    Parameters:
    cap (cv2.VideoCapture): Video capture object.
    num_frames (int): Number of frames to capture for the background.

    Returns:
    numpy.ndarray: The captured background frame.
    """
    background = None
    for _ in range(num_frames):
        ret, background = cap.read()
        if not ret:
            raise IOError("Cannot read frame from video capture")
    return background

def create_mask(hsv_frame):
    """
    Creates a mask for the red color in the frame.

    Parameters:
    hsv_frame (numpy.ndarray): The HSV color space frame.

    Returns:
    numpy.ndarray: The combined mask for the red color.
    """
    # Define the lower and upper bounds for the red color
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    # Create masks for the red color ranges
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

    # Combine the masks
    combined_mask = mask1 + mask2

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    return combined_mask

def apply_invisibility_effect(background, img, mask):
    """
    Applies the invisibility effect by masking the red color in the frame.

    Parameters:
    background (numpy.ndarray): The background frame.
    img (numpy.ndarray): The current frame.
    mask (numpy.ndarray): The mask for the red color.

    Returns:
    numpy.ndarray: The final frame with the invisibility effect applied.
    """
    # Invert the mask to get the non-red areas
    mask_inverted = cv2.bitwise_not(mask)

    # Segment out the red parts from the background
    res1 = cv2.bitwise_and(background, background, mask=mask)

    # Segment out the non-red parts from the current frame
    res2 = cv2.bitwise_and(img, img, mask=mask_inverted)

    # Combine the two results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    return final_output

def main():
    """
    Main function to execute the invisibility cloak effect.
    """
    # Open the video capture (webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Allow the camera to warm up
    time.sleep(2)

    # Capture the background frame
    background = capture_background(cap)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create the mask for the red color
        mask = create_mask(hsv)

        # Apply the invisibility effect
        final_output = apply_invisibility_effect(background, img, mask)

        # Display the final output
        cv2.imshow('Invisible Cloak', final_output)

        # Exit if the escape key is pressed
        if cv2.waitKey(2) == 27:
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
