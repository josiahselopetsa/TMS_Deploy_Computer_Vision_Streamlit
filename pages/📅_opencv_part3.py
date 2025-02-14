import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import tempfile
import dlib
from PIL import Image
import math
import scipy.signal

#______________________________________________________________________________________________

st.set_page_config(page_title="Opecv on Streamlit", page_icon="📅",layout="wide")
st.sidebar.markdown("### Opencv Part 3 📅")
#______________________________________________________________________________________________

# Function to read and process an image
def read_and_process_image(image_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_rgb = cv2.imdecode(file_bytes, 1)

    # # Read image file as binary
    # image_data = image_file.read()
    # # Convert to bytearray and then to numpy array
    # image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    # # Decode the numpy array as an image
    # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # # Check if the image was loaded successfully
    # if image is None:
    #     raise ValueError("Image decoding failed, image is None.")
    # # Convert from BGR (OpenCV default) to RGB (Streamlit expects)
    # # image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


# Function to display images using matplotlib
def display_image(img,tab, title=None):
    # plt.figure(figsize=(8, 6))
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plt.axis('off')
    fig, ax = plt.subplots()
    ax.imshow(image)
    if title:
        plt.title(title)
    tab.pyplot(fig)

# Streamlit UI components
st.title("OpenCV with Streamlit Part 3")
tab_Original,tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["original","1","2","3","4","5","6","7","8","9","10"])  # ,"G","R","Hue"

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and process the image
        image_rgb = read_and_process_image(uploaded_file)

        # Display the image using Streamlit
        tab_Original.image(image_rgb, caption="Processed Image", use_container_width=True, channels="BGR")
#______________________________________________________________________________________________
        Filter = tab1.selectbox("Filtering Colors",["bitwise_and", "bitwise_and2"])
        Filter_options = tab1.expander("Filtering Colors")
        placeholder1 = tab1.empty()
        image_Filter1 = image_rgb.copy()

        # Convert to Grayscale
        gray = cv2.cvtColor(image_Filter1, cv2.COLOR_BGR2GRAY)


        if Filter == "bitwise_and":
            # define range of BLUE color in HSV
            lower = np.array([90, 0, 0])
            upper = np.array([135, 255, 255])

            # Convert image from RBG/BGR to HSV so we easily filter
            hsv_img = cv2.cvtColor(image_Filter1, cv2.COLOR_BGR2HSV)

            # Use inRange to capture only the values between lower & upper
            mask = cv2.inRange(hsv_img, lower, upper)

            # Perform Bitwise AND on mask and our original frame
            image_Filter1 = cv2.bitwise_and(image_Filter1, image_Filter1, mask=mask)

            Filter_options.image(mask, caption = "mask")

        if Filter == "bitwise_and2":

            img_hsv = cv2.cvtColor(image_Filter1, cv2.COLOR_BGR2HSV)

            # lower mask (0-10)
            lower_red = np.array([0, 0, 0])
            upper_red = np.array([10, 255, 255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170, 0, 0])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join masks
            mask = mask0 + mask1

            # Perform Bitwise AND on mask and our original frame
            image_Filter1 = cv2.bitwise_and(image_Filter1, image_Filter1, mask=mask)
            Filter_options.image(mask, caption="mask")


        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_Filter1, cv2.COLOR_RGB2BGR)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder1.pyplot(fig)

# ______________________________________________________________________________________________
        Watershed = tab2.selectbox("Watershed algorithm for marker-based image segmentation", ["threshold", "distanceTransform"])
        Watershed_options = tab2.expander("Watershed algorithm for marker-based image segmentation")
        placeholder2 = tab2.empty()
        placeholder22 = tab2.empty()
        image_Watershed = image_rgb.copy()

        if Watershed == "threshold":
            # Grayscale
            gray = cv2.cvtColor(image_Watershed, cv2.COLOR_BGR2GRAY)

            # Threshold using OTSU
            ret, image_Watershed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if Watershed == "distanceTransform":
            # Grayscale
            gray = cv2.cvtColor(image_Watershed, cv2.COLOR_BGR2GRAY)
            # Threshold using OTSU
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            tab2.write("Finding unknown region")

            sure_fg = np.uint8(sure_fg)
            image_Watershed = cv2.subtract(sure_bg, sure_fg)

            Watershed_options.image(sure_fg, caption="SureFG")
            Watershed_options.image(sure_bg, caption="SureBG")



        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_Watershed, cv2.COLOR_RGB2BGR)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder2.pyplot(fig)

        # ______________________________________________________________________________________________

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.sidebar.write("Please upload an image file.")
# ______________________________________________________________________________________________
Foreground = tab3.selectbox("Background and Foreground Subtraction", ["foreground_background", "morphologyEx"])
Foreground_options = tab3.expander("Background and Foreground Subtraction")
placeholder3 = tab3.empty()
# image_Foreground = image_rgb.copy()

if Foreground == "foreground_background":

    tab3.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file = tab3.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # Get the height and width of the frame (required to be an interger)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/walking_output_foreground_background_part1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

        # Initlaize background subtractor
        foreground_background = cv2.createBackgroundSubtractorMOG2()
        # foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = tab3.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply background subtractor to get our foreground mask
                foreground_mask = foreground_background.apply(frame)
                out.write(cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR))

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(foreground_mask, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                Foreground_options.image(frame, channels="GRAY")

            cap.release()
            out.release()
            tab3.subheader('Video successful created')
            tab3.write("video path: /pages/output/walking_output_foreground_background_part1.avi")

if Foreground == "morphologyEx":

    tab3.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file = tab3.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # Get the height and width of the frame (required to be an interger)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/walking_output_morphologyEx_part1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.createBackgroundSubtractorKNN()

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = tab3.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                fgmask = fgbg.apply(frame)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                out.write(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                Foreground_options.image(frame, channels="GRAY")

            cap.release()
            out.release()
            tab3.subheader('Video successful created')
            tab3.write("video path: /pages/output/walking_output_morphologyEx_part1.avi")

# ______________________________________________________________________________________________
Meanshif = tab4.selectbox("Meanshif Object Tracking", ["car_tracking_mean_shift", "car_tracking_mean_shift2"])
Meanshif_options = tab4.expander("Meanshif Object Tracking")
placeholder4 = tab4.empty()
# image_Meanshif = image_rgb.copy()

if Meanshif == "car_tracking_mean_shift":

    tab4.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file4 = tab4.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file4 is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file4.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # take first frame of the video
        ret, frame = cap.read()

        # Get the height and width of the frame (required to be an interger)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/walking_output_GM_part1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

        # setup initial location of window
        r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = tab4.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                # apply meanshift to get the new location
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)

                # Draw it on image
                x,y,w,h = track_window
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255),2)
                out.write(img2)
                #imshow('Tracking', img2)

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                frame_placeholder.image(frame, channels="GRAY")

            cap.release()
            out.release()
            tab4.subheader('Video successful created')
            tab4.write("video path: /pages/output/walking_output_GM_part1.avi")

if Meanshif == "car_tracking_mean_shift2":

    tab4.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file4 = tab4.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file4 is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file4.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # take first frame of the video
        ret, frame = cap.read()

        # Get the height and width of the frame (required to be an interger)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/walking_output_GM_part2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

        # setup initial location of window
        r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
        track_window = (c, r, w, h)

        # set up the ROI for tracking
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = tab4.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                # Draw it on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, 255, 2)

                out.write(img2)
                # imshow('img2',img2)

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                frame_placeholder.image(frame, channels="GRAY")

            cap.release()
            out.release()
            tab4.subheader('Video successful created')
            tab4.write("video path: /pages/output/walking_output_GM_part2.avi")

# ______________________________________________________________________________________________
Lucas_Kanade = tab5.selectbox("The Lucas-Kanade Optical Flow Algorithm", ["Structure from Motion", "Video Compression"])
Lucas_Kanade_options = tab5.expander("The Lucas-Kanade Optical Flow Algorithm")
placeholder5 = tab5.empty()
# image_Lucas_Kanade = image_rgb.copy()

if Lucas_Kanade == "Structure from Motion":

    tab5.subheader('Lucas-Kanade Optical Flow with Streamlit')
    tab5.write('This app demonstrates Lucas-Kanade optical flow on an uploaded video.')

    # Function to process video and yield frames
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/Lucas_Kanade_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (width, height))

        # Set parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Set parameters for lucas kanade optical flow
        lucas_kanade_params = dict(winSize=(15, 15),
                                   maxLevel=2,
                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        # Used to create our trails for object movement in the image
        color = np.random.randint(0, 255, (100, 3))

        # Take first frame and find corners in it
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Find initial corner locations
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(prev_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                                   frame_gray,
                                                                   prev_corners,
                                                                   None,
                                                                   **lucas_kanade_params)

            # Select and store good points
            good_new = new_corners[status == 1]
            good_old = prev_corners[status == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv2.add(frame, mask)

            # Save Video
            out.write(img)

            # Convert frame to PIL image
            pil_img = Image.fromarray(img)

            yield pil_img

            # Now update the previous frame and previous points
            prev_gray = frame_gray.copy()
            prev_corners = good_new.reshape(-1, 1, 2)

        cap.release()
        out.release()


    # File uploader for video files
    uploaded_file = tab5.file_uploader("Choose a video file to view results", type=["avi", "mp4", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            temp_video_path = temp_video.name

        # Process and display video frames
        frame_iterator = process_video(temp_video_path)

        # Display frames using Streamlit
        for frame in frame_iterator:
            placeholder5.image(frame, caption='Lucas-Kanade Optical Flow', use_container_width=True)

        tab5.subheader('Video successful created')
        tab5.text("video path: /pages/output/Lucas_Kanade_optical_flow_walking.avi")

if Lucas_Kanade == "Video Compression":

    tab5.subheader('Dense Optical Flow with Streamlit')
    tab5.write('This app demonstrates dense optical flow on an uploaded video.')

    # Function to process video and yield frames
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/dense_optical_flow_walking.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (width, height))

        # Get first frame
        ret, first_frame = cap.read()
        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(first_frame)
        hsv[..., 1] = 255

        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
            next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Compute the dense optical flow using Farneback's algorithm
            flow = cv2.calcOpticalFlowFarneback(previous_gray, next_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * (180 / np.pi / 2)
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Save Video
            out.write(final)

            # Convert frame to PIL image
            pil_img = Image.fromarray(final)

            yield pil_img

            # Store current image as previous image
            previous_gray = next_gray

        cap.release()
        out.release()

    # File uploader for video files
    uploaded_file = tab5.file_uploader("Choose a video file", type=["avi", "mp4", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            temp_video_path = temp_video.name

        # Process and display video frames
        frame_iterator = process_video(temp_video_path)

        # Display frames using Streamlit
        for frame in frame_iterator:
            placeholder5.image(frame, caption='Dense Optical Flow', use_container_width=True)

        tab5.subheader('Video successful created')
        tab5.text("video path: /pages/output/dense_optical_flow_walking.avi")

# ______________________________________________________________________________________________
HSV_Filter = tab6.selectbox("Simple Object Tracking by Color", ["HSV Color Filter to Create a Mask and then Track our Desired Object"])
HSV_Filter_options = tab6.expander("Simple Object Tracking by Color")
placeholder6 = tab6.empty()
# image_Lucas_Kanade = image_rgb.copy()

if HSV_Filter == "HSV Color Filter to Create a Mask and then Track our Desired Object":

    tab6.subheader('Object Tracking with HSV and Contours in Streamlit')
    # tab6.text('This app demonstrates object tracking using HSV color space and contours on an uploaded video.')


    # Function to process video and yield frames
    def process_video(video_path):
        # Define range of color in HSV
        lower = np.array([20, 50, 90])
        upper = np.array([40, 255, 255])

        # Create empty points array
        points = []

        # Load video stream
        cap = cv2.VideoCapture(video_path)

        # Get the height and width of the frame (required to be an integer)
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
        out = cv2.VideoWriter('./pages/output/bmwm4_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

        ret, frame = cap.read()
        Height, Width = frame.shape[:2]
        frame_count = 0
        radius = 0

        while ret:
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv_img, lower, upper)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create empty centre array to store centroid center of mass
            center = int(Height / 2), int(Width / 2)

            if contours:
                # Get the largest contour and its center
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)

                # Sometimes small contours of a point will cause a division by zero error
                try:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                except ZeroDivisionError:
                    center = int(Height / 2), int(Width / 2)

                # Allow only contours that have a larger than 25 pixel radius
                if radius > 25:
                    # Draw circle and leave the last center creating a trail
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)

                # Log center points
                points.append(center)

            # If radius large enough, we use 25 pixels
            if radius > 25:
                # Loop over the set of tracked points
                for i in range(1, len(points)):
                    try:
                        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
                    except:
                        pass

            # Save Video
            out.write(frame)

            # Convert frame to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            yield pil_img

            # Read next frame
            ret, frame = cap.read()

        cap.release()
        out.release()


    # File uploader for video files
    uploaded_file = tab6.file_uploader("Choose a video file", type=["avi", "mp4", "mov", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            temp_video_path = temp_video.name

        # tab6.text("Processing video...")

        # Process and display video frames
        frame_iterator = process_video(temp_video_path)

        # Display frames using Streamlit
        for frame in frame_iterator:
            placeholder6.image(frame, caption='Object Tracking', use_container_width=True)

        # tab6.text('Video processing complete. Processed video saved as bmwm4_output.avi')
        tab6.subheader('Video successful created')
        tab6.text("video path: /pages/output/bmwm4_output.avi")

#______________________________________________________________________________________________
Facial = tab7.selectbox("Facial Landmark Detection with Dlib",["get_frontal_face_detector"])
Facial_options = tab7.expander("Facial Landmark Detection with Dlib")
placeholder7 = tab7.empty()
#
# PREDICTOR_PATH = "./pages/OtherModels/shape_predictor_68_face_landmarks.dat"
# predictor = dlib.shape_predictor(PREDICTOR_PATH)
# # predictor = dlib.shape_predictor(image_Facial)
# detector = dlib.get_frontal_face_detector()


if Facial == "get_frontal_face_detector":

    # Define exceptions
    class TooManyFaces(Exception):
        pass


    class NoFaces(Exception):
        pass


    # Load dlib's face detector and facial landmark predictor
    PREDICTOR_PATH = "./pages/OtherModels/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    detector = dlib.get_frontal_face_detector()


    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im


    tab7.subheader('Facial Landmark Detection with Streamlit')
    tab7.text('This app detects facial landmarks on an uploaded image.')

    # File uploader for image files
    uploaded_file = tab7.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            temp_image.write(uploaded_file.getbuffer())
            temp_image_path = temp_image.name

        # tab7.text("Processing image...")

        # Read the image using OpenCV
        image = cv2.imread(temp_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

        try:
            # Get landmarks
            landmarks = get_landmarks(image)

            # Annotate landmarks on the image
            image_with_landmarks = annotate_landmarks(image, landmarks)
            image_with_landmarks_rgb = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)

            # Convert the result image to PIL format for displaying in Streamlit
            result_image_pil = Image.fromarray(image_with_landmarks_rgb)

            # Display the original and the result image
            Facial_options.image(image_rgb, caption='Original Image', use_container_width=True)
            tab7.image(result_image_pil, caption='Image with Landmarks', use_container_width=True)

        except TooManyFaces:
            tab7.error("Too many faces detected in the image. Please upload an image with only one face.")
        except NoFaces:
            tab7.error("No faces detected in the image. Please upload a different image.")
        except Exception as e:
            tab7.error(f"An error occurred: {e}")


#______________________________________________________________________________________________
swapped = tab8.selectbox("Performing Face Swaps",["swapped"])
swapped_options = tab8.expander("Performing Face Swaps")
placeholder8 = tab8.empty()

if swapped == "swapped":

    import streamlit as st
    import cv2
    import dlib
    import numpy as np
    from PIL import Image
    import tempfile


    # Define exceptions
    class TooManyFaces(Exception):
        pass


    class NoFaces(Exception):
        pass


    # Load dlib's face detector and facial landmark predictor
    PREDICTOR_PATH = "./pages/OtherModels/shape_predictor_68_face_landmarks.dat"
    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11

    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
        NOSE_POINTS + MOUTH_POINTS,
    ]

    COLOUR_CORRECT_BLUR_FRAC = 0.6

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im


    def draw_convex_hull(im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)


    def get_face_mask(im, landmarks):
        im = np.zeros(im.shape[:2], dtype=np.float64)

        for group in OVERLAY_POINTS:
            draw_convex_hull(im, landmarks[group], color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im


    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])


    def read_im_and_landmarks(image):
        im = image
        im = cv2.resize(im, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
        s = get_landmarks(im)

        return im, s


    def warp_im(im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        return output_im


    def correct_colours(im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)).astype(np.uint8)


    def swappy(image1, image2):
        im1, landmarks1 = read_im_and_landmarks(image1)
        im2, landmarks2 = read_im_and_landmarks(image2)

        M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return output_im.astype(np.uint8)


    tab8.subheader('Face Swap App')
    tab8.text('Upload two images and see the magic!')

    uploaded_file1 = tab8.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])
    uploaded_file2 = tab8.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image1:
            temp_image1.write(uploaded_file1.getbuffer())
            temp_image_path1 = temp_image1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image2:
            temp_image2.write(uploaded_file2.getbuffer())
            temp_image_path2 = temp_image2.name

        image1 = cv2.imread(temp_image_path1)
        image2 = cv2.imread(temp_image_path2)

        try:
            swapped_image1 = swappy(image1, image2)
            swapped_image2 = swappy(image2, image1)

            tab8.image(cv2.cvtColor(swapped_image1, cv2.COLOR_BGR2RGB), caption='Swapped Image 1', use_container_width=True)
            tab8.image(cv2.cvtColor(swapped_image2, cv2.COLOR_BGR2RGB), caption='Swapped Image 2', use_container_width=True)

        except TooManyFaces:
            tab8.error("Too many faces detected in one of the images. Please upload images with only one face each.")
        except NoFaces:
            tab8.error("No faces detected in one of the images. Please upload images with clear faces.")
        except Exception as e:
            tab8.error(f"An error occurred: {e}")

# ______________________________________________________________________________________________
#______________________________________________________________________________________________
Tilt_Shift = tab9.selectbox("Our Functions to implement Tilt Shift",["Tilt_Shift"])
Tilt_Shift_options = tab9.expander("Our Functions to implement Tilt Shift")
placeholder9 = tab9.empty()

# Function to read and process an image
def read_and_process_image2(uploaded_file):
    # Read image file as binary
    image_data = uploaded_file.read()
    # Convert to bytearray and then to numpy array
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    # Decode the numpy array as an image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image decoding failed, image is None.")
    # Convert from BGR (OpenCV default) to RGB (Streamlit expects)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

if Tilt_Shift == "Tilt_Shift":

    # Define your image processing functions here
    def generating_kernel(parameter):
        kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                           0.25, 0.25 - parameter / 2.0])
        return np.outer(kernel, kernel)


    def reduce_img(image):
        kernel = generating_kernel(0.4)
        output = scipy.signal.convolve2d(image, kernel, 'same')
        return output[:output.shape[0]:2, :output.shape[1]:2]


    def expand(image):
        kernel = generating_kernel(0.4)
        output = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
        output[:output.shape[0]:2, :output.shape[1]:2] = image
        output = scipy.signal.convolve2d(output, kernel, 'same') * 4
        return output


    def gauss_pyramid(image, levels):
        output = [image]
        for level in range(levels):
            output.append(reduce_img(output[level]))
        return output


    def lapl_pyramid(gauss_pyr):
        output = []
        for image1, image2 in zip(gauss_pyr[:-1], gauss_pyr[1:]):
            output.append(image1 - expand(image2)[:image1.shape[0], :image1.shape[1]])
        output.append(gauss_pyr[-1])
        return output


    def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
        blended_pyr = []
        for lapl_white, lapl_black, gauss_mask in zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
            blended_pyr.append(gauss_mask * lapl_white + (1 - gauss_mask) * lapl_black)
        return blended_pyr


    def collapse(pyramid):
        output = pyramid[-1]
        for image in reversed(pyramid[:-1]):
            output = image + expand(output)[:image.shape[0], :image.shape[1]]
        return output


    def run_blend(black_image, white_image, mask):
        min_size = min(black_image.shape)
        depth = int(math.floor(math.log(min_size, 2))) - 4

        gauss_pyr_mask = gauss_pyramid(mask, depth)
        gauss_pyr_black = gauss_pyramid(black_image, depth)
        gauss_pyr_white = gauss_pyramid(white_image, depth)

        lapl_pyr_black = lapl_pyramid(gauss_pyr_black)
        lapl_pyr_white = lapl_pyramid(gauss_pyr_white)

        outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
        outimg = collapse(outpyr)

        outimg[outimg < 0] = 0
        outimg[outimg > 255] = 255
        outimg = outimg.astype(np.uint8)

        return outimg


    # Streamlit UI setup
    tab9.title('Image Blending with Gaussian and Laplacian Pyramids')

    # Upload images
    uploaded_black_img = tab9.file_uploader("Choose the original image...", type=["jpg", "jpeg", "png"], key="black")
    uploaded_white_img = tab9.file_uploader("Choose the blurred image...", type=["jpg", "jpeg", "png"], key="white")
    uploaded_mask_img = tab9.file_uploader("Choose the mask image...", type=["jpg", "jpeg", "png"], key="mask")

    if uploaded_black_img and uploaded_white_img and uploaded_mask_img:
        black_img = read_and_process_image2(uploaded_black_img)
        white_img = read_and_process_image2(uploaded_white_img)
        mask_img = read_and_process_image2(uploaded_mask_img)

        assert black_img.shape == white_img.shape, "Error - the sizes of original and blurred images are not equal"
        assert black_img.shape == mask_img.shape, "Error - the sizes of original and mask images are not equal"

        Tilt_Shift_options.image(black_img, caption='Original Image', use_container_width=True)
        Tilt_Shift_options.image(white_img, caption='Blurred Image', use_container_width=True)
        Tilt_Shift_options.image(mask_img, caption='Mask Image', use_container_width=True)

        tab9.write("Applying blending...")
        black_img = black_img.astype(float)
        white_img = white_img.astype(float)
        mask_img = mask_img.astype(float) / 255

        out_layers = []
        for channel in range(3):
            outimg = run_blend(black_img[:, :, channel], white_img[:, :, channel], mask_img[:, :, channel])
            out_layers.append(outimg)

        outimg = cv2.merge(out_layers)
        tab9.image(outimg, caption='Blended Image', use_container_width=True)
        tab9.write('...[DONE]')
#______________________________________________________________________________________________
GrabCut = tab10.selectbox("GrabCut Algorithm for background Removal",["grabCut"])
GrabCut_options = tab10.expander("GrabCut Algorithm for background Removal")
placeholder10 = tab10.empty()

tab10.subheader("GrabCut Segmentation using OpenCV")

def read_and_process_image(image_file):
    # Read the image file to a numpy array
    image = np.array(Image.open(image_file))
    return image

# File uploader
uploaded_file = tab10.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    image = read_and_process_image(uploaded_file)
    copy = image.copy()

    # Create a mask (of zeros uint8 datatype) that is the same size (width, height) as our original image
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Needs to be set Manually or selected with cv2.selectROI()
    x1, y1, x2, y2 = 190, 70, 350, 310
    start = (x1, y1)
    end = (x2, y2)

    # Format is X,Y,W,H
    rect = (x1, y1, x2 - x1, y2 - y1)

    # Show Rectangle
    cv2.rectangle(copy, start, end, (0, 0, 255), 3)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]

    # Display the original image with rectangle
    GrabCut_options.image(copy, caption='Original Image with Rectangle', use_container_width=True)

    # Display the segmented image
    placeholder10.image(segmented_image, caption='Segmented Image', use_container_width=True)
