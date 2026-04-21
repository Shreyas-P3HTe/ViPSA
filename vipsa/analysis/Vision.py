import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from typing import Optional, Tuple
import numpy as np
import math

class Camera : 
    """Class to handle camera operations.
        This class provides methods to capture images from the camera, process them, and display them in a GUI.
    
    Methods:
        - start_camera(): Initializes the camera and starts the GUI to display the camera feed.
        - capture_image(device_id): Captures a single image from the specified camera device.
        - get_thresholds(image_array): Applies Otsu's thresholding to the input image and returns the thresholded image.
        - get_contours(image_array, debug=False): Detects contours in the input image and returns the filtered contours and their scores.
        - get_contour_distances(image_array, filtered_contours, show_image=True): Calculates the distances of the detected contours from the center of the image and returns the distances along with an image showing the contours.
        - detect_movement_with_crop(crop_coords, threshold=1, min_area=2): Detects movement in a specified cropped region of the camera feed and displays the results in real-time.
        - overlay(image): Draws an X mark and angle lines on the input image for visualization purposes.
        - draw_x_rot(frame): Draws an X mark on the input frame for visualization purposes.
    """

    def overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw X mark and angle lines in the bottom-right quadrant.
        Parameters:
            image (np.ndarray): The input image on which to draw the marks.
        Returns:
            np.ndarray: The image with the marks drawn.
        """

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotate the frame by 180 degrees
        #image = cv2.rotate(image, cv2.ROTATE_180)

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

    def draw_x_rot(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw an X mark on the input frame for visualization purposes.
        Parameters:
            frame (np.ndarray): The input image/frame on which to draw the X mark.
        Returns:
            np.ndarray: The image/frame with the X mark drawn.
        """
        height, width, _ = frame.shape
        color = (0, 0, 255)
        thickness = 2
        cv2.line(frame, (0, 0), (width, height), color, thickness)
        cv2.line(frame, (width, 0), (0, height), color, thickness)
        return frame

    def start_camera(self) -> None:
        """Initializes the camera and starts the GUI to display the camera feed.
        This method sets up a Tkinter GUI to display the live feed from the camera.
        It also provides functionality to capture photos and save them to a specified folder.
        Arguments: None
        Returns: None"""

        def update_frame()-> None:
            """Capture a frame from the camera, process it, and update the GUI.
            This function is called repeatedly to update the video feed in the GUI.
            Arguments: None
            Returns: None"""

            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read frame.")
                root.destroy()
                return

            frame = draw_x_rot(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Convert to PIL format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update GUI
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            # Keep reference for saving
            video_label.current_frame = frame

            root.after(20, update_frame)

        def browse_folder()-> None:
            """Open a folder selection dialog and update the folder path variable."""

            folder = filedialog.askdirectory()
            if folder:
                folder_var.set(folder)

        def capture_photo()-> None:
            """Capture the current frame and save it to the specified folder with the given filename.
             This function retrieves the current frame from the video feed and saves it as a PNG file in the selected folder.
             Arguments: None
             Returns: None"""
            
            filename = filename_var.get()
            folder = folder_var.get()
            if not folder:
                messagebox.showerror("Error", "Please choose a folder to save the image.")
                return
            if not filename.lower().endswith(".png"):
                filename += ".png"
            filepath = os.path.join(folder, filename)
            frame_to_save = getattr(video_label, "current_frame", None)
            if frame_to_save is not None:
                cv2.imwrite(filepath, frame_to_save)
                messagebox.showinfo("Photo Saved", f"Photo saved as {filepath}")

        def close_window()-> None:
            """Release the camera and close the GUI window."""
            cap.release()
            root.destroy()

        # Tkinter GUI setup
        root = tk.Tk()
        root.title("Camera Viewer")
        root.geometry("+800+400")

        video_label = tk.Label(root)
        video_label.pack()

        # Input and browse
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(pady=10)

        filename_var = tk.StringVar(value="photo.png")
        folder_var = tk.StringVar()

        tk.Entry(bottom_frame, textvariable=filename_var, width=25).grid(row=0, column=0, padx=5)
        tk.Button(bottom_frame, text="Browse", command=browse_folder).grid(row=0, column=1, padx=5)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Capture Photo", width=15, command=capture_photo).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Exit", width=15, command=close_window).grid(row=0, column=1, padx=10)

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            root.destroy()
            return

        update_frame()
        root.protocol("WM_DELETE_WINDOW", close_window)
        root.mainloop()

    def capture_image(device_id) -> Optional[np.ndarray]:
        """Captures a single image from the specified camera device.
        This function initializes the camera, captures a frame, processes it by cropping and rotating,
        and then returns the processed image.
        Arguments: device_id (int): The ID of the camera device to capture from.
        Returns: np.ndarray: The processed image.
        """

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

    def get_thresholds(image_array) -> np.ndarray:
        """Applies Otsu's thresholding to the input image and returns the thresholded image.
        This function takes an image as input, converts it to grayscale, applies Gaussian blur to reduce noise,
        and then applies Otsu's thresholding to segment the image. The resulting binary image is returned.
        Arguments: image_array (np.ndarray): The input image to be processed.
        Returns: np.ndarray: The thresholded binary image.
        
        Note : This is critical for the contour detection step, as it helps to isolate the features of interest
        (e.g., pads) from the background. However, a cleverer approach to thresholding and contour detection is 
        needed to ensure that the correct features are identified, especiall in the case where the contrast between
        the pads and the background is not very high. The current implementation may need to be refined to improve 
        accuracy and robustness.
         """
        
        # Convert to grayscale
        gray_image = image_array
        
        # Apply Gaussian blur (optional but helps in better thresholding)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Apply Otsu's thresholding
        # Use cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV based on your needs
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        #plt.imshow(thresholded_image)
        
        return thresholded_image

    def get_contours(image_array, debug=False) -> Tuple[list, np.ndarray] | Tuple[list, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Detects contours in the input image and returns the filtered contours and their scores.
        This function processes the input image to detect contours that correspond to the features of interest 
        (e.g., pads). 
        
        It applies several image processing techniques such as cropping, contrast enhancement, blurring, 
        adaptive thresholding, and morphological operations to improve contour detection.
        
        ONE MODIFICATION : We have ditched Otsu's thresholding in favor of a more tailored approach that includes cropping around the center,
        contrast enhancement, and morphological operations to better isolate the features of interest.
        """

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2

        # ---- 1) crop only around the center ----
        roi_half_w = 170
        roi_half_h = 140

        x1 = max(0, cx - roi_half_w)
        x2 = min(w, cx + roi_half_w)
        y1 = max(0, cy - roi_half_h)
        y2 = min(h, cy + roi_half_h)

        roi = gray[y1:y2, x1:x2]

        # ---- 2) improve local contrast ----
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        roi_eq = clahe.apply(roi)

        # ---- 3) blur lightly ----
        roi_blur = cv2.GaussianBlur(roi_eq, (5, 5), 0)

        # ---- 4) adaptive threshold for bright pads on dark background ----
        thresh = cv2.adaptiveThreshold(
            roi_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            -3
        )

        # ---- 5) morphology cleanup ----
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = []
        scores = []

        roi_h, roi_w = roi.shape
        roi_cx, roi_cy = roi_w / 2, roi_h / 2
        margin = 6

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 8000:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Reject shapes touching ROI border
            if x <= margin or y <= margin or (x + bw) >= (roi_w - margin) or (y + bh) >= (roi_h - margin):
                continue

            aspect = bw / float(bh)
            if not (0.7 <= aspect <= 1.3):
                continue

            rect_area = bw * bh
            extent = area / rect_area if rect_area > 0 else 0
            if extent < 0.55:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.85:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

            dist2 = (cX - roi_cx) ** 2 + (cY - roi_cy) ** 2
            square_penalty = abs(1.0 - aspect) * 1000
            score = dist2 + square_penalty

            # shift contour back to full-image coordinates
            cnt_global = cnt + np.array([[[x1, y1]]])

            filtered.append(cnt_global)
            scores.append(score)

        if filtered:
            filtered = [cnt for _, cnt in sorted(zip(scores, filtered), key=lambda z: z[0])]

        if debug:
            debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for i, cnt in enumerate(contours):
                cv2.drawContours(debug_img, [cnt], -1, (100, 100, 100), 1)
            return filtered, thresh, roi, (x1, y1, x2, y2)

        return filtered, np.array(scores)
        
    def get_contour_distances(image_array, filtered_contours, show_image=True):
        center_x = image_array.shape[1] // 2
        center_y = image_array.shape[0] // 2

        image_with_contours = image_array.copy()

        if filtered_contours is None or len(filtered_contours) == 0:
            cv2.putText(image_with_contours, "NO PAD", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return np.array([None]), np.array([None]), image_with_contours

        best = filtered_contours[0]
        M = cv2.moments(best)

        if M["m00"] == 0:
            cv2.putText(image_with_contours, "BAD CONTOUR", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return np.array([None]), np.array([None]), image_with_contours

        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]

        x_distance = cX - center_x
        y_distance = cY - center_y

        if show_image:
            cv2.drawContours(image_with_contours, [best], -1, (0, 255, 0), 2)
            cv2.circle(image_with_contours, (int(cX), int(cY)), 4, (255, 0, 0), -1)
            cv2.putText(image_with_contours, "1", (int(cX) + 8, int(cY) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return np.array([x_distance]), np.array([y_distance]), image_with_contours

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

    def _get_contour_distances_old(image_array, filtered_contours, show_image=True):
        center_x = image_array.shape[1] // 2
        center_y = image_array.shape[0] // 2

        if not filtered_contours:
            return np.array([None]), np.array([None]), image_array.copy()

        best_contour = filtered_contours[0]  # already center-ranked in get_contours()

        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return np.array([None]), np.array([None]), image_array.copy()

        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]

        x_distance = cX - center_x
        y_distance = cY - center_y

        image_with_contours = image_array.copy()
        if show_image:
            cv2.drawContours(image_with_contours, [best_contour], -1, (0, 255, 0), 2)
            cv2.putText(
                image_with_contours,
                "1",
                (int(cX), int(cY)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2
            )

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


_DEFAULT_CAMERA = Camera()


def overlay(image: np.ndarray) -> np.ndarray:
    """Module-level compatibility wrapper for ``Camera.overlay``."""
    return _DEFAULT_CAMERA.overlay(image)


def draw_x_rot(frame: np.ndarray) -> np.ndarray:
    """Module-level compatibility wrapper for ``Camera.draw_x_rot``."""
    return _DEFAULT_CAMERA.draw_x_rot(frame)


def capture_image(device_id=0) -> Optional[np.ndarray]:
    """Module-level compatibility wrapper for ``Camera.capture_image``."""
    return Camera.capture_image(device_id)


def get_thresholds(image_array) -> np.ndarray:
    """Module-level compatibility wrapper for ``Camera.get_thresholds``."""
    return Camera.get_thresholds(image_array)


def get_contours(image_array, debug=False):
    """Module-level compatibility wrapper for ``Camera.get_contours``."""
    return Camera.get_contours(image_array, debug=debug)


def get_contour_distances(image_array, filtered_contours, show_image=True):
    """Module-level compatibility wrapper for ``Camera.get_contour_distances``."""
    return Camera.get_contour_distances(image_array, filtered_contours, show_image=show_image)


def detect_movement_with_crop(crop_coords, threshold=1, min_area=2):
    """Module-level compatibility wrapper for ``Camera.detect_movement_with_crop``."""
    return Camera.detect_movement_with_crop(crop_coords, threshold=threshold, min_area=min_area)

