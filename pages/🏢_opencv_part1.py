import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.filters import threshold_local

#______________________________________________________________________________________________

st.set_page_config(page_title="Opecv on Streamlit", page_icon="🏢",layout="wide")
st.sidebar.markdown("### Opencv Part 1 🏢")
#______________________________________________________________________________________________
# Function to read and process an image
def read_and_process_image(image_file):
    # Read image file as binary
    image_data = image_file.read()
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

# Function to display images using matplotlib
def display_image(img,tab, title=None):
    # plt.figure(figsize=(8, 6))
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.axis('off')
    fig, ax = plt.subplots()
    ax.imshow(image)
    if title:
        plt.title(title)
    tab.pyplot(fig)

# Streamlit UI components
st.title("OpenCV with Streamlit Part 1")

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["1","2","3","4","5","6","7","8","9","10"])  # ,"G","R","Hue"

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and process the image
        image_rgb = read_and_process_image(uploaded_file)

        # Display the image using Streamlit
        tab1.image(image_rgb, caption="Original Image", use_container_width=True)
#______________________________________________________________________________________________
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Example processing: Convert to grayscale
        display_image(image, tab2, title='Gray view')

        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image, cmap='gray')
        # plt.title("Gray view")
        # # ax.axis('on')  # Hide axes
        # tab2.pyplot(fig)
# ______________________________________________________________________________________________
        selection = tab3.selectbox("Select",["Split","Red","Green","Blue","Amplify","HSV"])
        # Use cv2.split to get each color space separately
        B, G, R = cv2.split(image_rgb)
        zeros = np.zeros(image.shape[:2], dtype="uint8")

        if selection == "Split":
            merges = B
        if selection ==  "Red":
            merges = cv2.merge([zeros, zeros, R])
        if selection ==  "Green":
            merges = cv2.merge([zeros, G, zeros])
        if selection ==  "Blue":
            merges = cv2.merge([B, zeros, zeros])
        if selection ==  "Amplify":
            amplify1 = tab3.slider("Blue amplifier",1,255)
            amplify2 = tab3.slider("Green amplifier", 1, 255)
            amplify3 = tab3.slider("Red amplifier", 1, 255)
            merges = cv2.merge([B+amplify1, G+amplify2, R+amplify3])
        if selection == "HSV":
            image_HSV = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
            selection_HSV = tab3.selectbox("Select", ["Hue", "Saturation", "Value"])
            if selection_HSV == "Hue":
                merges = image_HSV[:, :, 0]
            if selection_HSV == "Saturation":
                merges = image_HSV[:, :, 1]
            if selection_HSV == "Value":
                merges = image_HSV[:, :, 2]

        display_image(merges, tab3, title='Color Spaces')

#         # Example processing: Convert to grayscale
#         image = cv2.cvtColor(merges, cv2.COLOR_RGB2BGR)
#         # Plotting with Matplotlib
#         fig, ax = plt.subplots()
#         # ax.imshow(image, cmap='gray')
#         ax.imshow(image)
#         # ax.axis('on')  # Hide axes
#         tab3.pyplot(fig)
# # ______________________________________________________________________________________________
        draw = tab4.selectbox("Select", ["line", "rectangle", "circle", "polylines", "putText"])
        adjustment = tab4.expander("Adjustments")
        # Create a black image using numpy to create and array of black
        image_black = np.zeros((512, 512, 3), np.uint8)

        start_point_V = adjustment.slider("Start point y-axis", 0,530)
        start_point_H = adjustment.slider("Start point x-axis", 0,530)
        end_point_V = adjustment.slider("End point y-axis", 0,530,450)
        end_point_H = adjustment.slider("End point x-axis", 0,530,450)

        color1 = adjustment.number_input("colour R", 0,255)
        color2 = adjustment.number_input("colour G", 0,255,255)
        color3 = adjustment.number_input("colour B", 0,255)

        thickness = adjustment.number_input("Thickness",1,10,3)
        radius = adjustment.slider("radius", 10, 300,10)
        fill = adjustment.select_slider("fill",[ -1,1,2,10,50,100])

        if draw == "line":
            cv2.line(image_black, (start_point_H, start_point_V), (end_point_H, end_point_V), (color1, color2, color3), thickness)
            # cv2.line(image_black, (0, 0), (511, 511), (255, 127, 0), 5)
        if draw == "rectangle":
            cv2.rectangle(image_black, (start_point_H, start_point_V), (end_point_H, end_point_V), (color1, color2, color3), thickness)
            # cv2.rectangle(image_black, (100,100), (300,250), (127,50,127), 10)
        if draw == "circle":
            cv2.circle(image_black, (start_point_H, start_point_V),radius, (color1, color2, color3), fill)
            # cv2.circle(image_black, (350, 350), 100, (15,150,50), -1)
        if draw == "polylines":

            # Let's define four points
            pts = np.array([[start_point_H, start_point_V], [400, 50], [90, 200], [end_point_H, end_point_V]], np.int32)
            # pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)
            # Let's now reshape our points in form  required by polylines
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image_black, [pts], True, (color1, color2, color3), thickness)
            # cv2.polylines(image_black, [pts], True, (0,0,255), 3)
        if draw == "putText":
            ourString = 'Hello World!'
            cv2.putText(image_black, ourString, (start_point_H, start_point_V), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3,(color1, color2, color3), thickness)
            # cv2.putText(image_black, ourString, (10,200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 10)

        display_image(image_black, tab4, title='Drawing images and shapes using OpenCV')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_black, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # tab4.pyplot(fig)
# ______________________________________________________________________________________________
        position = tab5.selectbox("position", ["translation", "rotation RotationMatrix", "rotation transpose", "flipping"])
        placeholder5 = tab5.empty()
        img =  cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        # Store height and width of the image
        height, width = image_rgb.shape[:2]

        if position == "translation":
            # We shift it by quarter of the height and width
            quarter_height, quarter_width = height / 4, width / 4
            # T is our translation matrix
            T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
            # We use warpAffine to transform the image using the matrix, T
            img = cv2.warpAffine(img, T, (width, height))
            # st.image(img,channels="RGB")
        elif position == "rotation RotationMatrix":
            # Divide by two to rototate the image around its centre
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
            # Input our image, the rotation matrix and our desired final width and height
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
        elif position == "rotation transpose":
            #  rototate
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif position == "flipping":
            # Let's now to a horizontal flip.
            img = cv2.flip(img, 1)

        display_image(img, tab5, title='Transformations - Translations and Rotations')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # # ax.axis('on')  # Hide axes
        # placeholder5.pyplot(fig)
# ______________________________________________________________________________________________
        position6 = tab6.selectbox("Resize",[ "smaller", "larger","original","cropping"])
        # Resize = tab6.expander("resize")

        image_scaled =  cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        if position6 == "smaller":
            image_scaled = cv2.pyrDown(image_scaled)

        elif position6 == "larger":
            image_scaled = cv2.pyrUp(image_scaled)

        elif position6 == "original":
            image_scaled2 =  image_rgb.copy()
            selection_Resize = tab6.selectbox("Select resize",["Interpolation", "Cubic", "Nearest", "Area"])
            if selection_Resize == "Interpolation":
                # Let's make our image 3/4 of it's original size
                image_scaled = cv2.resize(image_scaled2, None, fx=0.75, fy=0.75)
            if selection_Resize == "Cubic":
                # Let's double the size of our image
                image_scaled = cv2.resize(image_scaled2, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
            if selection_Resize == "Nearest":
                # Let's double the size of our image using inter_nearest interpolation
                image_scaled = cv2.resize(image_scaled2, None, fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
            if selection_Resize == "Area":
                # Let's skew the re-sizing by setting exact dimensions
                image_scaled = cv2.resize(image_scaled2, (900, 400), interpolation = cv2.INTER_AREA)


        elif position6 == "cropping":
            # Get our image dimensions
            height, width = image_rgb.shape[:2]
            # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
            # using 0.25 to get the x,y position that is 1/4 down from the top left (0,0)
            start_row, start_col = int(height * .25), int(width * .25)
            # Let's get the ending pixel coordinates (bottom right)
            end_row, end_col = int(height * .75), int(width * .75)
            # Simply use indexing to crop out the rectangle we desire
            image_scaled = image_scaled[start_row:end_row, start_col:end_col]

            copy = image_rgb.copy()
            cv2.rectangle(copy, (start_col, start_row), (end_col, end_row), (0, 255, 255), 10)

        display_image(image_scaled, tab6, title='Scaling, Re-sizing, Interpolations and Cropping')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_scaled, cv2.COLOR_RGB2BGR)
        #
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # tab6.pyplot(fig)

# ______________________________________________________________________________________________
        bitwise = tab7.selectbox("Arithmetic Bitwise",["Grayscaled", "Increasing Brightness", "Decreasing Brightness","bitwise_shapes"])
        options = tab7.expander("Other options")

        placeholder7 = tab7.empty()
        image_bitwise = image_rgb.copy()
        image_bitwise = cv2.cvtColor(image_bitwise, cv2.COLOR_BGR2GRAY)

        # Create a matrix of ones, then multiply it by a scaler of 100
        # This gives a matrix with same dimesions of our image with all values being 100
        M = np.ones(image_bitwise.shape, dtype="uint8") * 100
        # Making a shapes
        shapes_s = np.zeros((300, 300), np.uint8)
        square = cv2.rectangle(shapes_s, (50, 50), (250, 250), 255, -2)
        shapes_e = np.zeros((300, 300), np.uint8)
        ellipse = cv2.ellipse(shapes_e, (150, 150), (150, 150), 30, 0, 180, 255, -1)

        if bitwise == "Grayscaled":
            image_bitwise = image_bitwise
        if bitwise == "Increasing Brightness":
            bright = options.selectbox("Arithmetic Bitwise", ["Brightness1", "Brightness2"])
            if bright == "Brightness1":
                # We use this to add this matrix M, to our image
                # Notice the increase in brightness
                image_bitwise = cv2.add(image_bitwise, M)
            if bright == "Brightness2":
                # Now if we just added it, look what happens
                image_bitwise = image_bitwise + M

        if bitwise == "Decreasing Brightness":
            darkness = options.selectbox("Arithmetic Bitwise", ["Darkenss1", "Darkenss2"])
            if darkness == "Darkenss1":
                # Likewise we can also subtract
                # Notice the decrease in brightness
                image_bitwise = cv2.subtract(image_bitwise, M)
            if darkness == "Darkenss2":
                image_bitwise = image_bitwise - M

        if bitwise == "bitwise_shapes":
            bitwise_shapes = options.selectbox("Bitwise shapes", ["rectangle", "ellipse", "bitwise_and", "bitwise_or", "bitwise_xor"])
            if bitwise_shapes == "rectangle":
                image_bitwise = square
            if bitwise_shapes == "ellipse":
                image_bitwise = ellipse
            if bitwise_shapes == "bitwise_and":
                image_bitwise = cv2.bitwise_and(square, ellipse)
            if bitwise_shapes == "bitwise_or":
                image_bitwise = cv2.bitwise_or(square, ellipse)
            if bitwise_shapes == "bitwise_xor":
                image_bitwise = cv2.bitwise_xor(square, ellipse)

        display_image(image_bitwise, tab7, title='Arithmetic and Bitwise Operations')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_bitwise, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder7.pyplot(fig)

# ______________________________________________________________________________________________
        Convolution = tab8.selectbox("Convolution Operations",["Blurring", "Denoising", "Sharpening"])
        Convolution_options = tab8.expander("Other options")
        placeholder8 = tab8.empty()
        image_filter = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        if Convolution == "Blurring":
            options8 = Convolution_options.selectbox("Bitwise shapes", ["3x3", "7x7", "blur", "GaussianBlur", "medianBlur"])
            if options8 == "3x3":
                # Creating our 3 x 3 kernel
                kernel_3x3 = np.ones((3, 3), np.float32) / 9
                # We use the cv2.fitler2D to conovlve the kernal with an image
                image_filter = cv2.filter2D(image_filter, -1, kernel_3x3)
            if options8 == "7x7":
                # Creating our 7 x 7 kernel
                kernel_7x7 = np.ones((7, 7), np.float32) / 49
                # We use the cv2.fitler2D to conovlve the kernal with an image
                image_filter = cv2.filter2D(image_filter, -1, kernel_7x7)
            if options8 == "blur":
                # This takes the pixels under the box and replaces the central element
                # Box size needs to odd and positive
                image_filter = cv2.blur(image_filter, (5, 5))
            if options8 == "GaussianBlur":
                # Instead of box filter, gaussian kernel
                image_filter = cv2.GaussianBlur(image_filter, (5, 5), 0)
            if options8 == "medianBlur":
                # Takes median of all the pixels under kernel area and central
                # element is replaced with this median value
                image_filter = cv2.medianBlur(image_filter, 5)
            if options8 == "medianBlur":
                # Bilateral is very effective in noise removal while keeping edges sharp
                image_filter = cv2.bilateralFilter(image_filter, 9, 75, 75)

        if Convolution == "Denoising":
            image_filter = cv2.fastNlMeansDenoisingColored(image_filter, None, 6, 6, 7, 21)

        if Convolution == "Sharpening":
            # Create our shapening kernel, remember it must sum to one
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
            # applying the sharpening kernel to the image
            image_filter = cv2.filter2D(image_filter, -1, kernel_sharpening)

        display_image(image_filter, tab8, title='Convolutions, Blurring and Sharpening Images')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_filter, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder8.pyplot(fig)
# ______________________________________________________________________________________________
        thresholding = tab9.selectbox("Thresholding Methods",["Threshold", "adaptiveThreshold", "skimage filters"])
        thresholding_options = tab9.expander("Other options")
        placeholder9 = tab9.empty()
        image_threshold = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        if thresholding == "Threshold":
            options9 = thresholding_options.selectbox("Other options", ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"])
            if options9 == "THRESH_BINARY":
                # Values below 127 goes to 0 or black, everything above goes to 255 (white)
                ret, image_threshold = cv2.threshold(image_threshold, 127, 255, cv2.THRESH_BINARY)

            if options9 == "THRESH_BINARY_INV":
                # Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
                ret, image_threshold = cv2.threshold(image_threshold, 127, 255, cv2.THRESH_BINARY_INV)
            if options9 == "THRESH_TRUNC":
                # Values above 127 are truncated (held) at 127 (the 255 argument is unused)
                ret, image_threshold = cv2.threshold(image_threshold, 127, 255, cv2.THRESH_TRUNC)
            if options9 == "THRESH_TOZERO":
                # Values below 127 go to 0, above 127 are unchanged
                ret, image_threshold = cv2.threshold(image_threshold, 127, 255, cv2.THRESH_TOZERO)
            if options9 == "THRESH_TOZERO_INV":
                # Reverse of the above, below 127 is unchanged, above 127 goes to 0
                ret, image_threshold = cv2.threshold(image_threshold, 127, 255, cv2.THRESH_TOZERO_INV)

        if thresholding == "adaptiveThreshold":
            # Convert the image to grayscale
            gray = cv2.cvtColor(image_threshold, cv2.COLOR_BGR2GRAY)

            options9 = thresholding_options.selectbox("adaptive comparison", ["adaptiveThreshold", "GaussianBlur"])
            if options9 == "adaptiveThreshold":
                # Using adaptiveThreshold
                image_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
            if options9 == "GaussianBlur":
                # Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, image_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if thresholding == "skimage filters":
            # We get the Value component from the HSV color space
            # then we apply adaptive thresholdingto
            V = cv2.split(cv2.cvtColor(image_threshold, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 25, offset=15, method="gaussian")
            # Apply the threshold operation
            image_threshold = (V > T).astype("uint8") * 255

        display_image(image_threshold, tab9, title='Thresholding, Binarization & Adaptive Thresholding')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_threshold, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder9.pyplot(fig)
# ______________________________________________________________________________________________
        Edges = tab10.selectbox("Edges",["Dilation", "Erosion", "Opening", "Closing", "Canny Edge Detection"])

        placeholder10 = tab10.empty()
        image_Edges = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # Let's define our kernel size
        kernel = np.ones((5, 5), np.uint8)

        if Edges == "Dilation":
            # Dilate here
            image_Edges = cv2.dilate(image_Edges, kernel, iterations=1)
        if Edges == "Erosion":
            # Now we erode
            image_Edges = cv2.erode(image_Edges, kernel, iterations=1)
        if Edges == "Opening":
            # Opening - Good for removing noise
            image_Edges = cv2.morphologyEx(image_Edges, cv2.MORPH_OPEN, kernel)
        if Edges == "Closing":
            # Closing - Good for removing noise
            image_Edges = cv2.morphologyEx(image_Edges, cv2.MORPH_CLOSE, kernel)
        if Edges == "Canny Edge Detection":
            Edges_options = tab10.expander("Other options")
            thresholds = Edges_options.number_input("thresholds", 10, 500, 50)
            edges = Edges_options.number_input("edges", 10, 500, 120)
            # Canny Edge Detection uses gradient values as thresholds
            # The first threshold gradient
            image_Edges = cv2.Canny(image_Edges, thresholds, edges)
            # image_Edges = cv2.Canny(image_Edges, 50, 120)

        display_image(image_Edges, tab10, title='Dilation, Erosion and Edge Detection')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_Edges, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder10.pyplot(fig)
# ______________________________________________________________________________________________

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.sidebar.write("Please upload an image file.")

