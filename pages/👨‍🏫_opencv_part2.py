import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tempfile
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from PIL import Image
#______________________________________________________________________________________________

st.set_page_config(page_title="Opecv on Streamlit", page_icon="👨‍🏫",layout="wide")
st.sidebar.markdown("### Opencv Part 2 👨‍🏫")
#______________________________________________________________________________________________

# Function to read and process an image
def read_and_process_image(image_file):
    # Read the uploaded image
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


# Function to display images in Streamlit
def imshow(tabs, title, image, size=16):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    fig, ax = plt.subplots(figsize=(size * aspect_ratio, size))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    # ax.axis('off')
    tabs.pyplot(fig)


def plotColors(hist, centroids):
    # Create our blank barchart
    bar = np.zeros((100, 500, 3), dtype="uint8")

    x_start = 0
    # iterate over the percentage and dominant color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end = x_start + (percent * 500)
        cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),
                      color.astype("uint8").tolist(), -1)
        x_start = end
    return bar

# Streamlit UI components
st.title("OpenCV with Streamlit Part 2")
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
        Contours = tab1.selectbox("Contours lines",["findContours1", "findContours2", "findContours3", "findContours4","Drawing Contours", "Hierachy of Contours", "Contouring Modes"])
        Contours_options = tab1.expander("Gray with threshold and canny")
        placeholder1 = tab1.empty()
        # image_Contours1 = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        image_Contours1 = image_rgb.copy()
        # Convert to Grayscale
        gray = cv2.cvtColor(image_Contours1, cv2.COLOR_BGR2GRAY)


        if Contours == "findContours1":
            _, image_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Contours_options.image(image_thresh,caption="gray view")
            # Finding Contours
            # Use a copy of your image e.g. edged.copy(), since findContours alters the image
            contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        if Contours == "findContours2":
            # Canny Edges
            edged = cv2.Canny(gray, 30, 200)
            Contours_options.image(edged,caption="canny view")
            # Finding Contours
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        if Contours == "findContours3":
            _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Contours_options.image(th2,caption="After thresholding")

            # Use a copy of your image e.g. edged.copy(), since findContours alters the image
            contours, hierarchy = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        if Contours == "findContours4":
            _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Contours_options.image(th2,caption="After thresholding")

            # Use a copy of your image e.g. edged.copy(), since findContours alters the image
            contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)


        if Contours == "Drawing Contours":
            # Canny Edges
            edged = cv2.Canny(gray, 30, 200)
            Contours_options.image(edged,caption="After thresholding")
            # Finding Contours
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        if Contours == "Hierachy of Contours":
            _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Contours_options.image(th2,caption="After thresholding")

            # Use a copy of your image e.g. edged.copy(), since findContours alters the image
            contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        if Contours == "Contouring Modes":
            _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Contours_options.image(th2,caption="After thresholding")

            # Use a copy of your image e.g. edged.copy(), since findContours alters the image
            contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Draw all contours, note this overwrites the input image (inplace operation)
            # Use '-1' as the 3rd parameter to draw all
            cv2.drawContours(image_Contours1, contours, -1, (0, 255, 0), thickness=2)

        # Display result
        # display_image(image_Contours1,tab1, title='Matched Image with Rectangle')
        imshow(tab1, 'Matched Image with Rectangle', image_Contours1)
        # display_image(image_Contours1, tab1, title='Matched Image with Rectangle')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_Contours1, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder1.pyplot(fig)
# ______________________________________________________________________________________________
        Sort = tab2.selectbox("Sort out lines",["Contours lines", "sorted_contours", "approxPolyDP", "convexHull","matchShapes"])
        Sort_options = tab2.expander("Gray with threshold and canny")
        placeholder2 = tab2.empty()
        placeholder22 = tab2.empty()
        image_Sort = image_rgb.copy()
        # image_Sort = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Convert to Grayscale
        gray = cv2.cvtColor(image_Sort, cv2.COLOR_BGR2GRAY)

        # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # Find Canny edges
        edged2 = cv2.Canny(gray, 50, 200)

        Sort_options.image(edged2, caption="canny view")


        # Function we'll use to display contour area
        def get_contour_areas(contours):
            """returns the areas of all contours as list"""
            all_areas = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                all_areas.append(area)
            return all_areas

        # Functions we'll use for sorting by position
        def x_cord_contour(contours):
            """Returns the X cordinate for the contour centroid"""
            if cv2.contourArea(contours) > 10:
                M = cv2.moments(contours)
                return (int(M['m10'] / M['m00']))
            else:
                pass

        def label_contour_center(image, c):
            """Places a red circle on the centers of contours"""
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw the countour number on the image
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
            return image

        if Sort == "Contours lines":
            # Find contours and print how many were found
            # contours, hierarchy = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            Sort_options.write(f"Number of contours found {len(contours)}" )
            Sort_options.write(f"Contor Areas before sorting {get_contour_areas(contours)}" )

            # Sort contours large to small by area
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            Sort_options.write(f"Contour Areas after sorting...{get_contour_areas(sorted_contours)}" )
            # Iterate over our contours and draw one at a time
            for (i, c) in enumerate(sorted_contours):
                M = cv2.moments(c)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_Sort, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.drawContours(image_Sort, [c], -1, (0, 255, 0), 3)

            imshow(tab2, 'Moments, Sorting, Approximating & Matching Contours', image_Sort)

        if Sort == "sorted_contours":
            # Find contours and print how many were found
            # contours, hierarchy = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Computer Center of Mass or centroids and draw them on our image
            for (i, c) in enumerate(contours):
                orig = label_contour_center(image_Sort, c)
            # Sort by left to right using our x_cord_contour function
            contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

            # Labeling Contours left to right
            for (i, c) in enumerate(contours_left_to_right):
                cv2.drawContours(image_Sort, [c], -1, (0, 0, 255), 3)
                M = cv2.moments(c)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(image_Sort, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                (x, y, w, h) = cv2.boundingRect(c)

            imshow(tab2, 'Moments, Sorting, Approximating & Matching Contours', image_Sort)

        if Sort == "approxPolyDP":
            orig_image = image_rgb.copy()
            copy_image = image_rgb.copy()

            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            # Find contours
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Iterate through each contour
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(copy_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(orig_image, [c], 0, (0, 255, 0), 2)
            Sort_options.image(orig_image, caption="Drawing of Contours")
            Sort_options.image(copy_image, caption="Bounding Rectangles")
            # Iterate through each contour and compute the approx contour
            for c in contours:
                # Calculate accuracy as a percent of the contour perimeter
                accuracy = 0.03 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, accuracy, True)
                cv2.drawContours(image_Sort, [approx], 0, (0, 255, 0), 2)

            imshow(tab2, 'Moments, Sorting, Approximating & Matching Contours', image_Sort)

        if Sort == "convexHull":
            orig_image = image_rgb.copy()
            # Threshold the image
            ret, thresh = cv2.threshold(gray, 176, 255, 0)
            # Find contours
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_Sort, contours, 0, (0, 255, 0), 1)
            Sort_options.image(orig_image, caption="Drawing of Contours")
            # Sort Contors by area and then remove the largest frame contour
            n = len(contours) - 1
            contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
            # Iterate through contours and draw the convex hull
            for c in contours:
                hull = cv2.convexHull(c)
                cv2.drawContours(image_Sort, [hull], 0, (0, 255, 0), 2)
            tab2.write("Convex Hull")

            imshow(tab2, 'Moments, Sorting, Approximating & Matching Contours', image_Sort)

        if Sort == "matchShapes":

            # Image uploader for template and target images
            uploaded_template = tab2.file_uploader("Choose a template image...", type=["jpg", "jpeg", "png"])
            uploaded_target = tab2.file_uploader("Choose a target image...", type=["jpg", "jpeg", "png"])

            if uploaded_template and uploaded_target:
                # Read the uploaded template image
                template_bytes = np.asarray(bytearray(uploaded_template.read()), dtype=np.uint8)
                template = cv2.imdecode(template_bytes, 0)  # Load as grayscale

                # Read the uploaded target image
                target_bytes = np.asarray(bytearray(uploaded_target.read()), dtype=np.uint8)
                target = cv2.imdecode(target_bytes, 1)  # Load as color
                target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

                # Display template image
                tab2.subheader('Template Image')
                imshow(tab2,'Template', template)

                # Display target image
                tab2.subheader('Target Image')
                imshow(tab2,'Target', target)

                # Threshold both images
                ret, thresh1 = cv2.threshold(template, 127, 255, 0)
                ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

                # Find contours in template
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours by area and get the second largest contour (template contour)
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                if len(sorted_contours) > 1:
                    template_contour = sorted_contours[1]
                else:
                    tab2.write("Could not find a valid contour in the template image.")
                    tab2.stop()

                # Extract contours from target image
                contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                closest_contour = None
                min_match = float('inf')

                for c in contours:
                    # Iterate through each contour in the target image and compare contour shapes
                    match = cv2.matchShapes(template_contour, c, 3, 0.0)
                    if match < min_match:
                        min_match = match
                        closest_contour = c

                if closest_contour is not None:
                    cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
                    tab2.subheader('Output Image with Matched Contour')
                    imshow(tab2,'Output', target)
                else:
                    tab2.write("No matching contour found in the target image.")
            else:
                tab2.write("Please upload both the template and target images to continue.")

        # display_image(image_Sort, tab2, title='Moments, Sorting, Approximating & Matching Contours')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_Sort, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder2.pyplot(fig)
# ______________________________________________________________________________________________
        detect = tab3.selectbox("Line, Circle and Blob Detection",["Houghlines", "Probabilistic Houghlines", "Hough Circles", "Blob Detection"])
        Sort_options = tab3.expander("Line, Circle and Blob Detection")
        placeholder3 = tab3.empty()
        placeholder33 = tab3.empty()
        image_detect = image_rgb.copy()
        # image_detect = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Grayscale and Canny Edges extracted
        gray = cv2.cvtColor(image_detect, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 170, apertureSize=3)

        if detect == "Houghlines":
            # Our line threshold is set to 240 (number of points on line)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)

            # We iterate through each line and convert it to the format
            # required by cv2.lines (i.e. requiring end points)
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image_detect, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                tab3.write("Skipping risk")

        if detect == "Probabilistic Houghlines":
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 3, 25)
            tab3.write(lines.shape)

            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(image_detect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
        if detect == "Hough Circles":
            blur = cv2.medianBlur(gray, 5)

            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)

            # cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
            # cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 100)

            # circles = np.uint16(np.around(circles))
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(image_detect, (i[0], i[1]), i[2], (0, 0, 255), 5)

                    # draw the center of the circle
                    cv2.circle(image_detect, (i[0], i[1]), 2, (0, 0, 255), 8)
            else:
                tab3.write("No circles were detected.")
        #
        if detect == "Blob Detection":
            # Set up the detector with default parameters.
            detector = cv2.SimpleBlobDetector_create()

            # Detect blobs.
            keypoints = detector.detect(image_detect)

            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
            # the circle corresponds to the size of blob
            blank = np.zeros((1, 1))
            image_detect = cv2.drawKeypoints(image_detect, keypoints, blank, (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_detect, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image_detect, cv2.COLOR_BGR2RGB)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder3.pyplot(fig)
# ______________________________________________________________________________________________
        Blobs = tab4.selectbox("Counting Circles, Ellipses and Finding Waldo", ["SimpleBlobDetector_Params1", "SimpleBlobDetector_Params2","matchTemplate"])
        Blobs_options = tab4.expander("Counting Circles, Ellipses and Finding Waldo")
        placeholder4 = tab4.empty()
        image_Blobs = image_rgb.copy()
        image_Blobs1 = image_rgb.copy()

        if Blobs == "SimpleBlobDetector_Params1":
            # Intialize the detector using the default parameters
            detector = cv2.SimpleBlobDetector_create()

            # Detect blobs
            keypoints = detector.detect(image_Blobs)

            # Draw blobs on our image as red circles
            blank = np.zeros((1, 1))
            image_Blobs = cv2.drawKeypoints(image_Blobs, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            number_of_blobs = len(keypoints)
            text = "Total Number of Blobs: " + str(len(keypoints))
            cv2.putText(image_Blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

        if Blobs == "SimpleBlobDetector_Params2":
            # Set our filtering parameters
            # Initialize parameter settiing using cv2.SimpleBlobDetector
            params = cv2.SimpleBlobDetector_Params()

            # Set Area filtering parameters
            params.filterByArea = True
            params.minArea = 100

            # Set Circularity filtering parameters
            params.filterByCircularity = True
            params.minCircularity = 0.9

            # Set Convexity filtering parameters
            params.filterByConvexity = False
            params.minConvexity = 0.2

            # Set inertia filtering parameters
            params.filterByInertia = True
            params.minInertiaRatio = 0.01

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs
            keypoints = detector.detect(image_Blobs)

            # Draw blobs on our image as red circles
            blank = np.zeros((1, 1))
            image_Blobs = cv2.drawKeypoints(image_Blobs, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            number_of_blobs = len(keypoints)
            text = "Number of Circular Blobs: " + str(len(keypoints))
            cv2.putText(image_Blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        if Blobs == "matchTemplate":

            def find_template(main_image, template_image):
                # Convert the main image to grayscale
                gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

                # Match the template using cv2.matchTemplate
                result = cv2.matchTemplate(gray, template_image, cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Create Bounding Box
                top_left = max_loc
                bottom_right = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])
                cv2.rectangle(main_image, top_left, bottom_right, (0, 0, 255), 5)

                return main_image


            tab4.subheader("Where is Waldo?")
            tab4.write("Upload a main image and a template image to find the template within the main image.")

            uploaded_file_main = tab4.file_uploader("Choose the main image...", type=["jpg", "jpeg", "png"], key="main")
            uploaded_file_template = tab4.file_uploader("Choose the template image...", type=["jpg", "jpeg", "png"],
                                                      key="template")

            if uploaded_file_main is not None and uploaded_file_template is not None:
                main_image = Image.open(uploaded_file_main)
                template_image = Image.open(uploaded_file_template)

                main_image_np = np.array(main_image)
                template_image_np = np.array(template_image.convert("L"))  # Convert template to grayscale

                tab4.image(main_image, caption='Main Image.', use_container_width=True)
                tab4.image(template_image, caption='Template Image.', use_container_width=True)

                tab4.write("")
                tab4.write("Processing image...")

                # Find template in the main image
                result_image = find_template(main_image_np.copy(), template_image_np)

                tab4.image(result_image, caption='Detected Template.', use_container_width=True)

# ______________________________________________________________________________________________
        Corners = tab5.selectbox("Finding Corners", ["Harris Corners", "Good Features"])
        Blobs_options = tab5.expander("Finding Corners")
        placeholder5 = tab5.empty()
        image_Corners = image_rgb.copy()


        if Corners == "Harris Corners":
            gray = cv2.cvtColor(image_Corners, cv2.COLOR_BGR2GRAY)
            # The cornerHarris function requires the array datatype to be float32
            gray = np.float32(gray)
            harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)
            # We use dilation of the corner points to enlarge them\
            kernel = np.ones((7, 7), np.uint8)
            harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)
            # Threshold for an optimal value, it may vary depending on the image.
            image_Corners[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]

        if Corners == "Good Features":
            gray = cv2.cvtColor(image_Corners, cv2.COLOR_BGR2GRAY)
            # We specific the top 50 corners
            corners = cv2.goodFeaturesToTrack(gray, 150, 0.0005, 10)

            for corner in corners:
                x, y = corner[0]
                x = int(x)
                y = int(y)
                cv2.rectangle(image_Corners, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)


        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_Corners, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image_detect, cv2.COLOR_BGR2RGB)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder5.pyplot(fig)

# ______________________________________________________________________________________________
        Haarcascade = tab6.selectbox("Face and Eye Detection with HaarCascade Classifiers", ["detect faces", "detect faces and eyes"])
        Haarcascade_options = tab6.expander("Face and Eye Detection with HaarCascade Classifiers")
        placeholder6 = tab6.empty()
        placeholder66 = tab6.empty()
        image_Haarcascade = image_rgb

        if Haarcascade == "detect faces":
            # We point OpenCV's CascadeClassifier function to where our
            # classifier (XML file format) is stored
            face_classifier = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(image_Haarcascade, cv2.COLOR_BGR2GRAY)

            # Our classifier returns the ROI of the detected face as a tuple
            # It stores the top left coordinate and the bottom right coordiantes
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # When no faces detected, face_classifier returns and empty tuple
            if faces is ():
                print("No faces found")

            # We iterate through our faces array and draw a rectangle
            # over each face in faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image_Haarcascade, (x, y), (x + w, y + h), (127, 0, 255), 2)

        if Haarcascade == "detect faces and eyes":
            face_classifier = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_frontalface_default.xml')
            eye_classifier = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_eye.xml')

            gray = cv2.cvtColor(image_Haarcascade, cv2.COLOR_BGR2GRAY)

            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # When no faces detected, face_classifier returns and empty tuple
            if faces is ():
                print("No Face Found")

            for (x, y, w, h) in faces:
                cv2.rectangle(image_Haarcascade, (x, y), (x + w, y + h), (127, 0, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image_Haarcascade[y:y + h, x:x + w]
                eyes = eye_classifier.detectMultiScale(roi_gray, 1.2, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        # if Haarcascade == "detect faces and eyes":
        #     face_classifier = cv2.CascadeClassifier('./pages/haarcascades/haarcascade_frontalface_default.xml')
        #     eye_classifier = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_eye.xml')
        #
        #
        #     def face_detector(img):
        #         # Convert image to grayscale
        #         gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #         faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        #         if len(faces) == 0:
        #             return img
        #
        #         for (x, y, w, h) in faces:
        #             x = x - 50
        #             w = w + 50
        #             y = y - 50
        #             h = h + 50
        #             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #             roi_gray = gray[y:y + h, x:x + w]
        #             roi_color = img[y:y + h, x:x + w]
        #             eyes = eye_classifier.detectMultiScale(roi_gray)
        #
        #             for (ex, ey, ew, eh) in eyes:
        #                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
        #
        #         return img
        #
        #
        #     uploaded_file = tab6.file_uploader("Choose an image or video...",
        #                                      type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
        #
        #     if uploaded_file is not None:
        #         if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        #             image = Image.open(uploaded_file)
        #             tab6.image(image, caption='Uploaded Image.', use_container_width=True)
        #             tab6.write("")
        #             tab6.write("Detecting faces and eyes...")
        #
        #             detected_image = face_detector(np.array(image))
        #             tab6.image(detected_image, caption='Processed Image.', use_container_width=True)
        #
        #         elif uploaded_file.type in ["video/mp4", "video/mov", "video/avi"]:
        #             tfile = tempfile.NamedTemporaryFile(delete=False)
        #             tfile.write(uploaded_file.read())
        #
        #             cap = cv2.VideoCapture(tfile.name)
        #             stframe = tab6.empty()
        #
        #             while cap.isOpened():
        #                 ret, frame = cap.read()
        #                 if not ret:
        #                     break
        #
        #                 detected_frame = face_detector(frame)
        #                 stframe.image(detected_frame, channels="BGR", use_container_width=True)
        #
        #             cap.release()
        #             os.remove(tfile.name)

        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_Haarcascade, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image_detect, cv2.COLOR_BGR2RGB)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder6.pyplot(fig)

        # tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10
        # ______________________________________________________________________________________________

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.sidebar.write("Please upload an image file.")

# ______________________________________________________________________________________________
Vehicles = tab7.selectbox("Vehicle and Pedestrian Detection",["detect Pedestrians", "detect Vehicles or cars", "detect Vehicles or cars2"])
Vehicles_options = tab7.expander("Vehicle and Pedestrian Detection")
placeholder7 = tab7.empty()
placeholder77 = tab7.empty()
# image_Vehicles = image_rgb.copy()

if Vehicles == "detect Pedestrians":
    # Create our video capturing object

    # Set the title of the Streamlit app
    tab7.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file = tab7.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    # cap = cv2.VideoCapture('./pages/videos/cars.mp4')

    # Load our body classifier
    body_classifier = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_fullbody.xml')

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = tab7.empty()
            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Grayscale our image for faster processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Pass frame to our body classifier
                bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

                # Extract bounding boxes for any bodies identified
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Display the frame in the Streamlit app
                placeholder7.image(frame)
            # Release our video capture
            cap.release()
        # Remove the temporary file
        os.remove(temp_filename)

if Vehicles == "detect Vehicles or cars":
    # Create our video capturing object
    # cap = cv2.VideoCapture('cars.mp4')

    # Load our vehicle classifier
    vehicle_detector = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_car.xml')

    # Set the title of the Streamlit app
    tab7.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file = tab7.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # Check if the video file opened successfully
        if not cap.isOpened():
            tab7.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = st.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Grayscale our image for faster processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Pass frame to our body classifier
                vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

                # Extract bounding boxes for any bodies identified
                for (x, y, w, h) in vehicles:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                placeholder7.image(frame)

            # Release our video capture
            cap.release()
        # Remove the temporary file
        os.remove(temp_filename)

if Vehicles == "detect Vehicles or cars2":
    # Create our video capturing object
    cap = cv2.VideoCapture('cars.mp4')

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap.get(3))
    h = int(cap.get(4))

    output_file_path = './pages/output/cars_output.avi'
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

    vehicle_detector = cv2.CascadeClassifier('./pages/haarcascades/Haarcascades/haarcascade_car.xml')

    # Set the title of the Streamlit app
    tab7.subheader('Video Upload and Display')

    # File uploader widget
    uploaded_file = tab7.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(temp_filename)

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error('Error opening video file')
        else:
            # Set a placeholder for the video frame
            frame_placeholder = st.empty()

            # Read and display frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Pass frame to our body classifier
                vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

                # Extract bounding boxes for any bodies identified
                for (x, y, w, h) in vehicles:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Write the frame into the file 'output.avi'
                out.write(frame)

                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in the Streamlit app
                placeholder7.image(frame)


            cap.release()
            out.release()

            tab7.write("Processed Video")
            # tab7.video(output_file_path)

            # # Provide a download button for the processed video
            # with open(output_file_path, 'rb') as file:
            #     tab7.download_button(
            #         label="Download Processed Video",
            #         data=file,
            #         file_name="cars_output.avi",
            #         mime="video/avi"
            #     )

        # Remove the temporary file
        # os.remove(temp_filename)

# ______________________________________________________________________________________________
drawContours = tab8.selectbox("Perspective Transforms", ["getPerspectiveTransform"])
drawContours_options = tab8.expander("Perspective Transforms")
placeholder8 = tab8.empty()
placeholder88 = tab8.empty()
# image_drawContours = image_rgb


if drawContours == "getPerspectiveTransform":

    def find_document_contour(image):
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours
        cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)

        # Sort contours by area
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Loop over the contours to find the document contour
        for cnt in sorted_contours:
            # Approximate the contour
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

            if len(approx) == 4:
                return approx

        return None


    def perspective_transform(image, contour):
        # Our x, y coordinates of the four corners
        input_pts = np.float32(contour)

        # Desired output points
        output_pts = np.float32([[0, 0],
                                 [0, 800],
                                 [500, 800],
                                 [500, 0]])

        # Get our Transform Matrix, M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        # Apply the transform Matrix M using Warp Perspective
        dst = cv2.warpPerspective(image, M, (500, 800))

        return dst


    tab8.subheader("Document Perspective Transformation")
    tab8.write("Upload an image to find contours and apply a perspective transform.")

    uploaded_file = tab8.file_uploader("Choose one picture...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        tab8.image(image, caption='Uploaded Image.', use_container_width=True)

        tab8.write("")
        tab8.write("Processing image...")

        # Find the document contour
        contour = find_document_contour(image_np.copy())

        if contour is not None:
            tab8.write("Document contour found. Applying perspective transform...")
            transformed_image = perspective_transform(image_np, contour)

            tab8.image(transformed_image, caption='Transformed Image.', use_container_width=True)
        else:
            tab8.error("No document-like contour found. Please upload a different image.")
# ______________________________________________________________________________________________
Histogram = tab9.selectbox("Histogram Representations", ["calcHist", "centroidHistogram"])
Histogram_options = tab9.expander("Histogram Representations")
placeholder9 = tab9.empty()
placeholder99 = tab9.empty()
# image_Histogram = image_rgb.copy()

if Histogram == "calcHist":

    def plot_histogram(image):
        # Plotting the histogram for the entire image
        tab9.write("Grayscale Histogram")
        fig, ax = plt.subplots()
        ax.hist(image.ravel(), 256, [0, 256])
        tab9.pyplot(fig)

        # Plotting the histogram for each color channel
        tab9.write("Color Channel Histograms")
        color = ('b', 'g', 'r')
        fig, ax = plt.subplots()
        for i, col in enumerate(color):
            histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(histogram, color=col)
            ax.set_xlim([0, 256])
        tab9.pyplot(fig)


    tab9.subheader("Image Histogram with OpenCV")
    tab9.write("Upload an image to view its histogram.")

    uploaded_file = tab9.file_uploader("Choose an image calcHist ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        tab9.image(image, caption='Uploaded Image.', use_container_width=True)
        tab9.write("")
        tab9.write("Generating histogram...")

        plot_histogram(image_np)

if Histogram == "centroidHistogram":
    def centroidHistogram(clt):
        # Create a histogram for the clusters based on the pixels in each cluster
        # Get the labels for each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

        # Create our histogram
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # Normalize the histogram, so that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        return hist


    def plotColors(hist, centroids):
        # Create our blank bar chart
        bar = np.zeros((100, 500, 3), dtype="uint8")

        x_start = 0
        # Iterate over the percentage and dominant color of each cluster
        for (percent, color) in zip(hist, centroids):
            # Plot the relative percentage of each cluster
            end = x_start + (percent * 500)
            cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),
                          color.astype("uint8").tolist(), -1)
            x_start = end
        return bar


    tab9.subheader("Dominant Colors with K-means Clustering")
    tab9.write("Upload an image to find and display its dominant colors.")

    uploaded_file = tab9.file_uploader("Choose an image centroidHistogram ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        tab9.image(image, caption='Uploaded Image.', use_container_width=True)
        tab9.write("")
        tab9.write("Finding dominant colors...")

        # Convert image to RGB (OpenCV uses BGR by default)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_flattened = image_np.reshape((image_np.shape[0] * image_np.shape[1], 3))

        number_of_clusters = 5
        clt = KMeans(n_clusters=number_of_clusters)
        clt.fit(image_flattened)

        hist = centroidHistogram(clt)
        bar = plotColors(hist, clt.cluster_centers_)

        # Show the color bar chart
        tab9.write("Dominant Colors Bar Chart")
        tab9.image(bar, use_container_width=True)


# ______________________________________________________________________________________________
Comparing = tab10.selectbox("Comparing Images", ["Structual Similarity"])
Comparing_options = tab10.expander("Comparing Images")
placeholder10 = tab10.empty()
placeholder100 = tab10.empty()
# image_Comparing = image_rgb

def mse(image1, image2):
    # Images must be of the same dimension
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error

def compare_images(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_value = mse(image1_gray, image2_gray)
    ssim_value, _ = ssim(image1_gray, image2_gray, full=True)
    return mse_value, ssim_value

tab10.subheader("Image Comparison with MSE and SSIM")
tab10.write("Upload two images to compare them.")

uploaded_file1 = tab10.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"], key="image1")
uploaded_file2 = tab10.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"], key="image2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    image1_np = np.array(image1)
    image2_np = np.array(image2)

    Comparing_options.image(image1, caption='First Image.', use_container_width=True)
    Comparing_options.image(image2, caption='Second Image.', use_container_width=True)

    if image1_np.shape == image2_np.shape:
        tab10.write("")
        tab10.write("Comparing images...")

        mse_value, ssim_value = compare_images(image1_np, image2_np)
        tab10.write(f'MSE: {mse_value:.2f}')
        tab10.write(f'SSIM: {ssim_value:.2f}')
    else:
        tab10.error("Error: The two images must have the same dimensions.")

# Brightness adjustment example
if uploaded_file1 is not None:
    tab10.write("")
    tab10.write("Brightness Adjustment Example")

    image1_np = np.array(image1)
    M = np.ones(image1_np.shape, dtype="uint8") * 100
    image1_bright = cv2.add(image1_np, M)

    tab10.image(image1, caption='Original Image.', use_container_width=True)
    tab10.image(image1_bright, caption='Brightness Increased by 100.', use_container_width=True)


