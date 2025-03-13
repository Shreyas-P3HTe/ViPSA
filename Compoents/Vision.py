import PySimpleGUI as sg
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

import cv2
import numpy as np
import math

def overlay(image):
    """Draw X mark and angle lines in the bottom-right quadrant."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the frame by 180 degrees
    image = cv2.rotate(image, cv2.ROTATE_180)

    # Draw X mark in the center
    line_length = 5  # Length of the X lines
    cv2.line(image, (center[0] - line_length, center[1] - line_length),
             (center[0] + line_length, center[1] + line_length), (0, 0, 255), 1)
    cv2.line(image, (center[0] - line_length, center[1] + line_length),
             (center[0] + line_length, center[1] - line_length), (0, 0, 255), 1)
    
    line_length = 5  # Length of the X lines
    cv2.line(image, (center[0]-125 - line_length, center[1] - line_length),
             (center[0]-125 + line_length, center[1] + line_length), (0, 255, 0), 1)
    cv2.line(image, (center[0]-125 - line_length, center[1] + line_length),
             (center[0]-125 + line_length, center[1] - line_length), (0, 255, 0), 1)
    
    # Draw a grid centered around the image
    grid_spacing = 125
    line_color = (200, 200, 200)  # Light gray
    line_thickness = 1

    # Vertical lines
    for x in range(center[0] % grid_spacing, w, grid_spacing):
        cv2.line(image, (x, 0), (x, h), line_color, line_thickness)
    for x in range(center[0] % grid_spacing - grid_spacing, -1, -grid_spacing):
        cv2.line(image, (x, 0), (x, h), line_color, line_thickness)

    # Horizontal lines
    for y in range(center[1] % grid_spacing, h, grid_spacing):
        cv2.line(image, (0, y), (w, y), line_color, line_thickness)
    for y in range(center[1] % grid_spacing - grid_spacing, -1, -grid_spacing):
        cv2.line(image, (0, y), (w, y), line_color, line_thickness)

    # Draw faint grey angle lines in the bottom-right quadrant
    radius = min(w, h) // 2  # Define the maximum length of the lines
    angle_step = 5  # Angle step in degrees
    font_scale = 0.2
    font_color = (200, 200, 200)
    font_thickness = 1
    for angle in range(0, 91, angle_step):  # From 0° to 90° in 10° steps
        rad = math.radians(angle)
        end_x = int(center[0] + radius * math.cos(rad))
        end_y = int(center[1] + radius * math.sin(rad))

        # Draw the line
        cv2.line(image, center, (end_x, end_y), (200, 200, 200), 1, cv2.LINE_AA)

        # Add the angle label at the edges
        if angle == 90:  # Special cases for vertical and horizontal lines
            label_x = end_x - 20 if angle == 90 else end_x + 5
        else:
            label_x = end_x + 5
            label_y = end_y - 5

        cv2.putText(image, f"{angle}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return image

def start_camera():
    # Define the layout of the GUI
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.InputText(default_text="photo.png", size=(25, 1), key="-FILENAME-"), sg.FolderBrowse("Browse", key="-FOLDER-")],
        [sg.Button("Capture Photo", size=(10, 1)), sg.Button("Exit", size=(10, 1))]
    ]

    # Create the window
    window = sg.Window("Camera Viewer", layout, location=(800, 400))

    # Open a connection to the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sg.popup_error("Error: Could not open camera.")
        exit()

    while True:
        event, values = window.read(timeout=20)

        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            sg.popup_error("Error: Could not read frame.")
            break

        # Draw X mark on the frame
        processed_frame = draw_x_rot(frame)

        # Rotate the frame by 180 degrees
        rotated_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        imgbytes = cv2.imencode(".png", rotated_frame)[1].tobytes()

        # Update the image in the window
        window["-IMAGE-"].update(data=imgbytes)

        if event == "Capture Photo":
            # Get the filename from the input box
            filename = values["-FILENAME-"]
            
            # Check if a valid directory is chosen or not
            folder = values["-FOLDER-"]
            if not folder:
                sg.popup_error("Please choose a folder to save the image.")
                continue
            
            # Ensure the filename ends with '.png'
            if not filename.lower().endswith(".png"):
                filename += ".png"
            
            # Construct the full file path
            filepath = os.path.join(folder, filename)
            
            # Save the current frame as a file
            cv2.imwrite(filepath, rotated_frame)
            sg.popup("Photo saved as", filepath)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    # Release the camera and close the window
    cap.release()
    window.close()
    
def capture_image(device_id):
    # Initialize the camera
    camera = cv2.VideoCapture(device_id)

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return None

    # Capture a frame
    ret, frame = camera.read()

    if ret:
        # Get the dimensions of the frame
        #plt.imshow(frame)
        (h, w) = frame.shape[:2]

        # Define the amount to crop from each side
        crop_amount = 80  # Crop 50 pixels from each side
        startX, endX = crop_amount+50, w - (crop_amount+50)
        startY, endY = crop_amount, h - crop_amount

        # Crop the image equally from all sides
        cropped_image = frame[startY:endY, startX:endX]

        # Get new dimensions after cropping
        (ch, cw) = cropped_image.shape[:2]
        center = (cw // 2, ch // 2)

        # Get the rotation matrix for the cropped image
        rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)

        # Apply the rotation to the cropped image
        rotated_frame = cv2.warpAffine(cropped_image, rotation_matrix, (cw, ch))

        # Release the camera
        camera.release()

        return rotated_frame
    else:
        print("Error: Could not read frame.")
        camera.release()
        return None

def get_thresholds(image_array):
    
    # Convert to grayscale
    gray_image = image_array
    
    # Apply Gaussian blur (optional but helps in better thresholding)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    # Use cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV based on your needs
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #plt.imshow(thresholded_image)
    
    return thresholded_image

def get_contours(image_array):
    
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred_image, 50, 100, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #plt.imshow(thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_area = max(areas)

    filtered_contours = [cnt for cnt, area in zip(contours, areas) if area >= 0.1 * max_area]

    return filtered_contours, np.array(areas)
    
def get_contour_distances_old(image_array, filtered_contours, show_image=True):

    center_x = image_array.shape[1] // 2
    center_y = image_array.shape[0] // 2

    distances = []  # Initialize as a list
    ranks = []      # Initialize as a list

    for contour in filtered_contours:
        if len(contour) >= 10:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                distance = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)

                distances.append(distance)  # Append to the list
                ranks.append((cX - center_x)**2 + (cY - center_y)**2)  # Append to the list
    
    print(len(distances), len(ranks))
    sorted_contours = [contour for _, contour in sorted(zip(ranks, filtered_contours))]

    x_distances = [(cv2.moments(cnt)["m10"] / cv2.moments(cnt)["m00"]) - center_x for cnt in sorted_contours]
    y_distances = [(cv2.moments(cnt)["m01"] / cv2.moments(cnt)["m00"]) - center_y for cnt in sorted_contours]

    if show_image:

        image_with_contours = cv2.drawContours(image_array.copy(), sorted_contours, -1, (0, 255, 0), 2)
        for i, (x, y) in enumerate(zip(x_distances, y_distances)):
            cv2.putText(image_with_contours, f"{i+1}", (int(center_x + x), int(center_y + y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    return np.array(x_distances), np.array(y_distances), image_with_contours

def get_contour_distances(image_array, filtered_contours, show_image=True):

    center_x = image_array.shape[1] // 2
    center_y = image_array.shape[0] // 2

    min_rank = float('inf')  # Initialize with a large value
    closest_contour = None

    # Find the closest contour to the center
    for contour in filtered_contours:
        if len(contour) >= 10:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                rank = (cX - center_x)**2 + (cY - center_y)**2
                
                if rank < min_rank:
                    min_rank = rank
                    closest_contour = contour

    if closest_contour is not None:
        # Calculate the x and y distances of the closest contour
        M = cv2.moments(closest_contour)
        x_distance = (M["m10"] / M["m00"]) - center_x
        y_distance = (M["m01"] / M["m00"]) - center_y

        if show_image:
            # Draw only the closest contour
            image_with_contours = cv2.drawContours(image_array.copy(), [closest_contour], -1, (0, 255, 0), 2)
            cv2.putText(image_with_contours, "1", (int(center_x + x_distance), int(center_y + y_distance)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        else:
            image_with_contours = image_array
    else:
        x_distance, y_distance, image_with_contours = None, None, image_array

    return np.array([x_distance]), np.array([y_distance]), image_with_contours

def detect_movement_with_crop(crop_coords, threshold=1, min_area=2):
    """
    Detect small movements in a cropped region of the camera feed.

    Parameters:
    - crop_coords: tuple, (x, y, width, height) defining the ROI to crop.
    - threshold: int, intensity threshold for frame difference.
    - min_area: int, minimum area of detected movement to consider significant.
    """
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # Change the device index if needed
    
    if not cap.isOpened():
        print("Failed to access the camera.")
        return
    
    x, y, w, h = crop_coords
    
    print("Press 'q' to quit.")
    
    # Read the first frame and crop
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from the camera.")
        return
    
    prev_frame = frame[y:y+h, x:x+w]
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    while True:
        # Capture the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the ROI
        cropped_frame = frame[y:y+h, x:x+w]

        # Convert to grayscale and blur
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute the absolute difference between the current frame and previous frame
        frame_delta = cv2.absdiff(prev_frame_gray, gray)
        _, thresh = cv2.threshold(frame_delta, threshold, 5, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            # Draw a rectangle around the detected movement
            (cx, cy, cw, ch) = cv2.boundingRect(contour)
            cv2.rectangle(cropped_frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

        # Show the cropped frame and threshold image
        cv2.imshow("Cropped Movement Detection", cropped_frame)
        cv2.imshow("Threshold", thresh)

        # Update the previous frame
        prev_frame_gray = gray.copy()

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

