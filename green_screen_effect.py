# Filename: green_screen_effect.py

import cv2
import numpy as np

def initialize_video_capture(video_path, scale):
    """
    Initialize video capture and get video frame dimensions.

    Parameters:
    video_path (str): Path to the video file.
    scale (float): Scale factor for resizing the frame.

    Returns:
    cap (cv2.VideoCapture): Video capture object.
    res (tuple): Resolution of the scaled frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Read the first frame from the video
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read frame from video")
    
    # Get the height and width of the frame
    h, w = frame.shape[:2]
    # Scale the height and width
    h = int(h * scale)
    w = int(w * scale)
    # Set the resolution as a tuple
    res = (w, h)
    
    # Reset video to the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, res

def initialize_video_writer(output_path, res, fps=30.0):
    """
    Initialize the video writer.

    Parameters:
    output_path (str): Path to save the output video.
    res (tuple): Resolution of the video.
    fps (float): Frames per second.

    Returns:
    out (cv2.VideoWriter): Video writer object.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, res)
    return out

def process_frame(frame, bg_image, res, margin=50):
    """
    Process a single frame to replace the background.

    Parameters:
    frame (numpy.ndarray): Input video frame.
    bg_image (numpy.ndarray): Background image.
    res (tuple): Resolution of the output frame.
    margin (int): Margin for color masking.

    Returns:
    result_out (numpy.ndarray): Frame with background replaced.
    crop (numpy.ndarray): Cropped frame.
    mask (numpy.ndarray): Mask used for cropping.
    """
    # Resize the frame and background image to the same resolution
    frame_resized = cv2.resize(frame, res)
    bg_resized = cv2.resize(bg_image, res)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    # Split the HSV channels
    h, s, v = cv2.split(hsv)

    # Find the most common color in the saturation channel
    # Get Uniques Colors and its Counts
    unique_colors, counts = np.unique(s, return_counts=True)
    # Sort through and Grab the most abundant unique color
    big_color = None
    biggest = -1
    for a in range(len(unique_colors)):
        if counts[a] > biggest:
            biggest = counts[a]
            big_color = int(unique_colors[a])

    # Create a mask for the most common color with some margin
    mask = cv2.inRange(s, big_color - margin, big_color + margin)
    # Dilate and blur the mask to smooth it
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Create a black image for the cropped frame
    crop = np.zeros_like(frame_resized)
    # Copy the parts of the frame that are not masked
    crop[mask == 255] = frame_resized[mask == 255]

    # Replace the background in the frame
    obj = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
    f_bg = frame_resized - obj
    f_bg = np.where(f_bg == 0, f_bg, bg_resized)
    result_out = f_bg + obj

    return result_out, crop, mask

def display_frames(img, mask, result_out):
    """
    Display frames in separate windows.

    Parameters:
    img (numpy.ndarray): Original frame.
    mask (numpy.ndarray): Mask used for cropping.
    result_out (numpy.ndarray): Frame with background replaced.
    """
    # Create windows to display the images
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # Resize the windows for better visibility
    cv2.resizeWindow("Image", 400, 300)
    cv2.resizeWindow("Mask", 400, 300)
    cv2.resizeWindow("Result", 400, 300)

    # Show the images in the created windows
    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result_out)

def process_video(input_video_path, bg_image_path, output_video_path, scale=0.5):
    """
    Process the entire video to replace its background.

    Parameters:
    input_video_path (str): Path to the input video file.
    bg_image_path (str): Path to the background image file.
    output_video_path (str): Path to save the output video.
    scale (float): Scale factor for resizing the frame.
    """
    # Initialize the video capture and get the resolution
    cap, res = initialize_video_capture(input_video_path, scale)
    
    # Read the background image
    bg_image = cv2.imread(bg_image_path)
    if bg_image is None:
        raise IOError(f"Cannot open background image: {bg_image_path}")

    # Initialize the video writer to save the processed video
    out = initialize_video_writer(output_video_path, res)

    # Process each frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to replace the background
        result_out, crop, mask = process_frame(frame, bg_image, res)
        
        # Display the original frame, mask, and processed frame
        display_frames(frame, mask, result_out)

        # Check if the escape key (Esc) is pressed to exit
        if cv2.waitKey(2) == 27:
            break

        # Write the cropped frame to the output video
        out.write(crop)

    # Release the video capture and writer resources
    cap.release()
    out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define the paths for input video, background image, and output video
    input_video_path = "Images_Videos/Cat_video.mp4"
    bg_image_path = "Images_Videos/Advanture.jpg"
    output_video_path = "test_vid.avi"
    
    # Process the video
    process_video(input_video_path, bg_image_path, output_video_path)
