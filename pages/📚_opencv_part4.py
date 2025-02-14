import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import tempfile
from easyocr import Reader
from skimage.filters import threshold_local
from PIL import Image
from barcode import EAN13
from barcode.writer import ImageWriter
from pytesseract import Output
import qrcode
from pyzbar.pyzbar import decode
import pandas as pd
import time
import os
import random
import face_recognition
#______________________________________________________________________________________________

st.set_page_config(page_title="Opecv on Streamlit", page_icon="📚",layout="wide")
st.sidebar.markdown("### Opencv Part 4 📚")
#______________________________________________________________________________________________

# Function to read and process an image
def read_and_process_image(image_file):
    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # image_rgb = cv2.imdecode(file_bytes, 1)

    # Read image file as binary
    image_data = image_file.read()
    # Convert to bytearray and then to numpy array
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    # image_array = np.frombuffer(image_data, np.uint8)
    # Decode the numpy array as an image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # image = cv2.imdecode(image_array, 1)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image decoding failed, image is None.")
    # Convert from BGR (OpenCV default) to RGB (Streamlit expects)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
st.title("OpenCV with Streamlit Part 4")
tab_Original,tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["original","1","2","3","4","5","6","7","8","9","10"])  # ,"G","R","Hue"

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read and process the image
        image_rgb = read_and_process_image(uploaded_file)

        # Display the image using Streamlit
        tab_Original.image(image_rgb, caption="Processed Image", use_container_width=True, channels="RBG")
#______________________________________________________________________________________________
        PyTesseract = tab1.selectbox("Few images using PyTesseract", ["PyTesseract Extracted1", "PyTesseract Extracted2", "PyTesseract Extracted3", "easyocr Extracted1", "easyocr Extracted2"])
        PyTesseract_options = tab1.expander("Few images using PyTesseract")
        placeholder1 = tab1.empty()
        image_PyTesseract = image_rgb.copy()

        pytesseract.pytesseract.tesseract_cmd = (
            'C:\Program Files\Tesseract-OCR\tesseract.exe'
        )

        if PyTesseract == "PyTesseract Extracted1":
            # imshow("Input Image", img)
            PyTesseract_options.image(image_PyTesseract, caption="Input Image")
            # Run our image through PyTesseract
            output_txt = pytesseract.image_to_string(image_PyTesseract)

            PyTesseract_options.write(f"PyTesseract Extracted: {output_txt}")

        if PyTesseract == "PyTesseract Extracted2":
            # We get the Value component from the HSV color space
            # then we apply adaptive thresholdingto
            V = cv2.split(cv2.cvtColor(image_PyTesseract, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 25, offset=15, method="gaussian")

            # Apply the threshold operation
            thresh = (V > T).astype("uint8") * 255
            # imshow("threshold_local", thresh, size=48)
            PyTesseract_options.image(thresh, caption="threshold_local")

            output_txt = pytesseract.image_to_string(thresh)
            # print("PyTesseract Extracted: {}".format(output_txt))
            PyTesseract_options.write(f"PyTesseract Extracted: {output_txt}")

        if PyTesseract == "PyTesseract Extracted3":
            # We get the Value component from the HSV color space
            # then we apply adaptive thresholdingto
            V = cv2.split(cv2.cvtColor(image_PyTesseract, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 25, offset=15, method="gaussian")
            # Apply the threshold operation
            thresh = (V > T).astype("uint8") * 255
            d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
            # print(d.keys())
            tab1.write(f"d keys: {d.keys()}")

            n_boxes = len(d['text'])

            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    image_PyTesseract = cv2.rectangle(image_PyTesseract, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # imshow('Output', image, size=12)
            PyTesseract_options.image(image_PyTesseract, caption="Output")

        if PyTesseract == "easyocr Extracted1":
            # # load the input image from disk
            # image = cv2.imread("whatsapp_conv.jpeg")
            # imshow("Original Image", image, size=12)
            PyTesseract_options.image(image_PyTesseract, caption="Original Image")

            # OCR the input image using EasyOCR
            tab1.write("Detecting and OCR'ing text from input image...")
            reader = Reader(['en'], gpu=False)

            ts = time.time()
            results = reader.readtext(image_PyTesseract)
            te = time.time()
            td = te - ts
            tab1.write(f'Completed in {td} seconds')

            all_text = []

            # iterate over our extracted text
            for (bbox, text, prob) in results:
                # display the OCR'd text and the associated probability of it being text
                # print(f" Probability of Text: {prob * 100:.3f}% OCR'd Text: {text}")
                PyTesseract_options.write(f" Probability of Text: {prob * 100:.3f}% OCR'd Text:  {text}")

                # get the bounding box coordinates
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                # Remove non-ASCII characters from the text so that
                # we can draw the box surrounding the text overlaid onto the original image
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                all_text.append(text)
                cv2.rectangle(image_PyTesseract, tl, br, (255, 0, 0), 2)
                cv2.putText(image_PyTesseract, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # show the output image
            # imshow("OCR'd Image", image, size=12)

        if PyTesseract == "easyocr Extracted2":

            def clean_text(text):
                # remove non-ASCII text so we can draw the text on the image
                return "".join([c if ord(c) < 128 else "" for c in text]).strip()


            # image = cv2.imread('Receipt-woolworth.jpg')

            reader = Reader(["en", "ar"], gpu=False)
            results = reader.readtext(image_PyTesseract)

            # loop over the results
            for (bbox, text, prob) in results:
                # display the OCR'd text and associated probability
                # print("[INFO] {:.4f}: {}".format(prob, text))
                PyTesseract_options.write(f"EASY OCR Extracted: {format(prob)}  {format(text)}")

                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                # clean text and draw the box surrounding the text along
                text = clean_text(text)
                cv2.rectangle(image_PyTesseract, tl, br, (0, 255, 0), 2)
                cv2.putText(image_PyTesseract, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Apply the threshold operation
            # thresh = (V > T).astype("uint8") * 255
            # imshow("EASY OCR", image)
            # print("EASY OCR Extracted: {}".format(text))
            PyTesseract_options.write(f"EASY OCR Extracted: {format(text)}")

        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_PyTesseract, cv2.COLOR_RGB2BGR)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder1.pyplot(fig)
# ______________________________________________________________________________________________
        Barcode = tab2.selectbox("Barcode Generation and Reading",["Barcodes Generation","QRCode Generation","QRCode Generation1","QRCode Generation2"])
        Barcode_options = tab2.expander("Barcode Generation and Reading")
        placeholder2 = tab2.empty()
        image_Barcode = image_rgb.copy()

        if Barcode == "Barcodes Generation":
            # Function to generate barcode and return image
            def generate_barcode(code):
                filename = "barcode.png"
                with open(filename, 'wb') as f:
                    EAN13(code, writer=ImageWriter()).write(f)
                return filename


            # Streamlit app
            tab2.title("Barcode Generator")

            # Input for barcode number
            barcode_number = tab2.text_input("Enter a 12-digit number for the barcode", "123456789102")

            # Generate and display barcode
            if len(barcode_number) == 12 and barcode_number.isdigit():
                barcode_file = generate_barcode(barcode_number)
                image = Image.open(barcode_file)
                tab2.image(image, caption='Generated Barcode')
            else:
                tab2.warning("Please enter a valid 12-digit number.")


        if Barcode == "QRCode Generation":
            # Function to generate QR code and return image file
            def generate_qr_code(data, filename="qrcode.png"):
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(data)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                img.save(filename)
                return filename


            # Streamlit app
            tab2.write("QR Code Generator")

            # Input for QR code data
            qr_data = tab2.text_input("Enter the data for the QR code", "https://www.opencv.org")

            # Generate and display QR code
            if qr_data:
                qr_file = generate_qr_code(qr_data)
                image = Image.open(qr_file)
                tab2.image(image, caption='Generated QR Code')
            else:
                tab2.warning("Please enter some data to generate the QR code.")
        if Barcode == "QRCode Generation1":
            # image = cv2.imread("1DwED.jpg")

            # Detect and decode the qrcode
            codes = decode(image_Barcode)

            # loop over the detected barcodes
            for bc in codes:
                # Get the rect coordiantes for our text placement
                (x, y, w, h) = bc.rect
                print(bc.polygon)
                pt1, pt2, pt3, pt4 = bc.polygon

                # Draw a bounding box over our detected QR code
                pts = np.array([[pt1.x, pt1.y], [pt2.x, pt2.y], [pt3.x, pt3.y], [pt4.x, pt4.y]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_Barcode, [pts], True, (0, 0, 255), 3)

                # extract the string info data and the type from our object
                barcode_text = bc.data.decode()
                barcode_type = bc.type

                # show our
                text = "{} ({})".format(barcode_text, barcode_type)
                cv2.putText(image_Barcode, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(image_Barcode, barcode_type, (x + w, y + h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                Barcode_options.write(f"QR Code revealed: {format(text)} ")
        if Barcode == "QRCode Generation2":
            # Detect and decode the qrcode
            barcodes = decode(image_Barcode)

            # loop over the detected barcodes
            for bc in barcodes:
                # Get the rect coordiantes for our text placement
                (x, y, w, h) = bc.rect
                cv2.rectangle(image_Barcode, (x, y), (x + w, y + h), (255, 0, 0), 3)

                # extract the string info data and the type from our object
                barcode_text = bc.data.decode()
                barcode_type = bc.type

                # show our
                text = "{} ({})".format(barcode_text, barcode_type)
                cv2.putText(image_Barcode, barcode_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(image_Barcode, barcode_type, (x + w, y + h - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                Barcode_options.write("Barcode revealed: {}".format(barcode_text))
                Barcode_options.write("Barcode revealed: {}".format(barcode_text))


        # Example processing: Convert to grayscale
        image = cv2.cvtColor(image_Barcode, cv2.COLOR_RGB2BGR)
        # Plotting with Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        # ax.axis('on')  # Hide axes
        placeholder2.pyplot(fig)
# ______________________________________________________________________________________________
        YOLOv3 = tab3.selectbox("YOLOv3 in using cv2.dnn.readNetFrom", ["getLayerNames","Starting Detections"])
        YOLOv3_options = tab3.expander("YOLOv3 in using cv2.dnn.readNetFrom")
        placeholder3 = tab3.empty()
        image_YOLOv3 = image_rgb.copy()

        if YOLOv3 == "getLayerNames":
            # Load the COCO class labels our YOLO model was trained on
            labelsPath = "./pages/YOLO/YOLO/yolo/coco.names"
            LABELS = open(labelsPath).read().strip().split("\n")

            # We now need to initialize a list of colors to represent each possible class label
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

            YOLOv3_options.write("Loading YOLO weights...")

            weights_path = "./pages/YOLO/YOLO/yolo/yolov3.weights"
            cfg_path = "./pages/YOLO/YOLO/yolo/yolov3.cfg"

            # Create our blob object
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

            # Set our backend
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            YOLOv3_options.write("Our YOLO Layers")
            ln = net.getLayerNames()

            # There are 254 Layers
            YOLOv3_options.write(f"{len(ln)}, {ln} ")

        if YOLOv3 == "Starting Detections":

            tab3.write("Starting Detections...")
            # Load YOLO model

            weights_path = "./pages/YOLO/YOLO/yolo/yolov3.weights"
            cfg_path = "./pages/YOLO/YOLO/yolo/yolov3.cfg"
            net = cv2.dnn.readNet(weights_path, cfg_path)

            labelsPath = "./pages/YOLO/YOLO/yolo/coco.names"
            with open(labelsPath, "r") as f:
                LABELS = f.read().strip().split("\n")

            # Initialize a list of colors to represent each possible class label
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


            # Function to perform YOLO detection
            def detect_objects(image):
                (H, W) = image.shape[:2]
                ln = net.getLayerNames()
                ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                layerOutputs = net.forward(ln)

                boxes = []
                confidences = []
                IDs = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > 0.75:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            IDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        color = [int(c) for c in COLORS[IDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return image


            # Streamlit app
            tab3.write("YOLO Object Detection")

            image_YOLOv3 = detect_objects(image_YOLOv3)

            # Display the output image
            tab3.image(image_YOLOv3, caption='Detected Objects', use_container_width=True, channels="BGR")

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_YOLOv3, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder3.pyplot(fig)
# ______________________________________________________________________________________________
        neuralStyle = tab4.selectbox("Neural Style Transfer with OpenCV", ["neuralStyleModel1","neuralStyleModel2 ECCV16","neuralStyleModel starry_night","neuralStyleModel starry_night2"])
        neuralStyle_options = tab4.expander("Neural Style Transfer with OpenCV")
        placeholder4 = tab4.empty()
        image_neuralStyle = image_rgb.copy()

        if neuralStyle == "neuralStyleModel1":
            # Define the model directory and image paths
            model_dir = "./pages/NeuralStyleTransfer/NeuralStyleTransfer/models/"
            style_dir = "./pages/NeuralStyleTransfer/NeuralStyleTransfer/art/"
            image_path = image_neuralStyle

            # Load all model file paths
            model_file_paths = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

            # Load the test image
            # img = cv2.imread(image_path)
            img = image_neuralStyle

            # Set the fixed height for resizing
            fixed_height = 640

            # Display the original image
            tab4.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_container_width=True, channels="BGR")

            # Loop through and apply each model style to the input image
            for i, model in enumerate(model_file_paths):
                # Print the model being used
                tab4.write(f"{i + 1}. Using Model: {model[:-3]}")

                # Load the style image
                style_image_path = os.path.join(style_dir, f"{model[:-3]}.jpg")
                style = cv2.imread(style_image_path)

                # Display the style image
                tab4.image(cv2.cvtColor(style, cv2.COLOR_BGR2RGB), caption=f'Style: {model[:-3]}',
                         use_container_width=True)

                # Load the neural style transfer model
                neuralStyleModel = cv2.dnn.readNetFromTorch(os.path.join(model_dir, model))

                # Resize the input image
                height, width = img.shape[:2]
                new_width = int((fixed_height / height) * width)
                resized_img = cv2.resize(img, (new_width, fixed_height), interpolation=cv2.INTER_AREA)

                # Create a blob from the image
                inp_blob = cv2.dnn.blobFromImage(resized_img, 1.0, (new_width, fixed_height),
                                                 (103.939, 116.779, 123.68), swapRB=False, crop=False)

                # Set the input blob for the neural style transfer model
                neuralStyleModel.setInput(inp_blob)

                # Perform a forward pass of the network
                output = neuralStyleModel.forward()

                # Reshape the output tensor, add back the mean subtraction, and reorder the channels
                output = output.reshape(3, output.shape[2], output.shape[3])
                output[0] += 103.939
                output[1] += 116.779
                output[2] += 123.68
                output = output / 255.0  # Normalize to [0.0, 1.0]
                output = np.clip(output, 0, 1)  # Ensure values are within [0, 1]
                output = output.transpose(1, 2, 0)

                # Display the final neural style transferred image
                tab4.image(output, caption=f'Neural Style Transfer: {model[:-3]}', use_container_width=True)


        if neuralStyle == "neuralStyleModel2 ECCV16":
            # Function to apply neural style transfer
            def apply_neural_style_transfer(model_path, img):
                # Load the neural style transfer model
                neuralStyleModel = cv2.dnn.readNetFromTorch(model_path)

                # Resize the input image
                height, width = img.shape[:2]
                new_width = int((fixed_height / height) * width)
                resized_img = cv2.resize(img, (new_width, fixed_height), interpolation=cv2.INTER_AREA)

                # Create a blob from the image
                inp_blob = cv2.dnn.blobFromImage(resized_img, 1.0, (new_width, fixed_height),
                                                 (103.939, 116.779, 123.68), swapRB=False, crop=False)

                # Set the input blob for the neural style transfer model
                neuralStyleModel.setInput(inp_blob)

                # Perform a forward pass of the network
                output = neuralStyleModel.forward()

                # Reshape the output tensor, add back the mean subtraction, and reorder the channels
                output = output.reshape(3, output.shape[2], output.shape[3])
                output[0] += 103.939
                output[1] += 116.779
                output[2] += 123.68
                output = output / 255.0  # Normalize to [0.0, 1.0]
                output = np.clip(output, 0, 1)  # Ensure values are within [0, 1]
                output = output.transpose(1, 2, 0)

                return output


            # Define the model directory and image paths
            model_dir = "./pages/NeuralStyleTransfer/NeuralStyleTransfer/models/ECCV16/"
            style_dir = "./pages/NeuralStyleTransfer/NeuralStyleTransfer/art/"
            image_path = image_neuralStyle

            # Load all model file paths
            model_file_paths = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

            # Load the test image
            # img = cv2.imread(image_path)
            img = image_path

            # Set the fixed height for resizing
            fixed_height = 640

            # Display the original image
            tab4.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_container_width=True)

            # Loop through and apply each model style to the input image
            for i, model in enumerate(model_file_paths):
                # Print the model being used
                tab4.write(f"{i + 1}. Using Model: {model[:-3]}")

                # Load the style image
                style_image_path = os.path.join(style_dir, f"{model[:-3]}.jpg")
                style = cv2.imread(style_image_path)

                # Display the style image
                tab4.image(cv2.cvtColor(style, cv2.COLOR_BGR2RGB), caption=f'Style: {model[:-3]}', use_container_width=True)

                # Apply neural style transfer
                output = apply_neural_style_transfer(os.path.join(model_dir, model), img)

                # Display the final neural style transferred image
                tab4.image(output, caption=f'Neural Style Transfer: {model[:-3]}', use_container_width=True)


        if neuralStyle == "neuralStyleModel starry_night":

            tab4.subheader('Video Upload and Display')

            # File uploader widget
            uploaded_file = tab4.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

            if uploaded_file is not None:
                # Create a temporary file to save the uploaded video
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_filename = temp_file.name


                # Load our t7 neural transfer models
                model_file_path = "./pages/NeuralStyleTransfer/NeuralStyleTransfer/models/ECCV16/starry_night.t7"

                # Load video stream, long clip
                # cap = cv2.VideoCapture('dj.mp4')
                # Open the video file using OpenCV
                cap = cv2.VideoCapture(temp_filename)

                # Get the height and width of the frame (required to be an interger)
                w = int(cap.get(3))
                h = int(cap.get(4))

                # Define the codec and create VideoWriter object. The output is stored in '*.avi' file.
                output_dir = './pages/output/'
                # Define the codec and create VideoWriter object
                output_path = os.path.join(output_dir, 'NST_Starry_Night.mp4')
                # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

                # Loop through and applying each model style our input image
                # for (i,model) in enumerate(model_file_paths):
                style = cv2.imread("./pages/NeuralStyleTransfer/NeuralStyleTransfer/art/starry_night.jpg")
                i = 0
                # Check if the video file opened successfully
                if not cap.isOpened():
                    st.error('Error opening video file')
                else:
                    # Set a placeholder for the video frame
                    frame_placeholder = tab4.empty()

                    # Read and display frames from the video
                    while cap.isOpened():
                        ret, img = cap.read()
                        if not ret:
                            break

                        i += 1
                        tab4.write("Completed {} Frame(s)".format(i))
                        # loading our neural style transfer model
                        neuralStyleModel = cv2.dnn.readNetFromTorch(model_file_path)

                        # Let's resize to a fixed height of 640 (feel free to change)
                        height, width = int(img.shape[0]), int(img.shape[1])
                        newWidth = int((640 / height) * width)
                        resizedImg = cv2.resize(img, (newWidth, 640), interpolation=cv2.INTER_AREA)

                        # Create our blob from the image and then perform a forward pass run of the network
                        inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640),
                                                        (103.939, 116.779, 123.68), swapRB=False, crop=False)

                        neuralStyleModel.setInput(inpBlob)
                        output = neuralStyleModel.forward()

                        # Reshaping the output tensor, adding back  the mean subtraction
                        # and re-ordering the channels
                        output = output.reshape(3, output.shape[2], output.shape[3])
                        output[0] += 103.939
                        output[1] += 116.779
                        output[2] += 123.68
                        output /= 255
                        output = np.clip(output, 0, 1)
                        output = output.transpose(1, 2, 0)

                        # Display our original image, the style being applied and the final Neural Style Transfer
                        # imshow("Original", img)
                        # imshow("Style", style)
                        # imshow("Neural Style Transfers", output)
                        vid_output = (output * 255).astype(np.uint8)
                        vid_output = cv2.resize(vid_output, (w, h), interpolation=cv2.INTER_AREA)
                        out.write(vid_output)

                    cap.release()
                    out.release()

        if neuralStyle == "neuralStyleModel starry_night2":
            # Display the processed video in Streamlit

            # video_bytes = open('./pages/videos/walking.mp4','rb').read()
            # video_bytes = open('./pages/output/NST_Starry_Night.mp4', 'rb').read()
            # st.video(video_bytes)
            tab4.write('Neural Style Transfer on Video')
            # tab4.video(video_bytes)
            tab4.video('./pages/output/NST_Starry_Night.mp4')

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_neuralStyle, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder4.pyplot(fig)
# ______________________________________________________________________________________________
        SSDs = tab5.selectbox("Single Shot Detectors (SSDs) with OpenCV", ["ssd_mobilenet_v1_coco"])
        SSDs_options = tab5.expander("Single Shot Detectors (SSDs) with OpenCV")
        placeholder5 = tab5.empty()
        image_SSDs = image_rgb.copy()

        if SSDs == "ssd_mobilenet_v1_coco":
            # Load the pre-trained model
            prototxt = "./pages/SSDs/SSDs/ssd_mobilenet_v1_coco.pbtxt"
            weights = "./pages/SSDs/SSDs/frozen_inference_graph.pb"
            net = cv2.dnn.readNetFromTensorflow(weights, prototxt)

            # Set the widths and heights that are needed for input into our model
            inWidth = 300
            inHeight = 300
            WHRatio = inWidth / float(inHeight)

            # These are needed for our preprocessing of our image
            inScaleFactor = 0.007843
            meanVal = 127.5

            # Number of classes
            num_classes = 90

            # Probability Threshold
            thr = 0.5

            # Class names
            classNames = {
                0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                14: 'parking meter',
                15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee',
                35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard',
                42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
                58: 'hot dog',
                59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                67: 'dining table',
                70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
                78: 'microwave',
                79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
                87: 'scissors',
                88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
            }

            frame = image_SSDs

            # Create our input image blob required for input into our network
            blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal),
                                         swapRB=True)
            net.setInput(blob)

            # Pass our input image/blob into the network
            detections = net.forward()

            # Crop frame if needed as we don't resize our input but take a square input
            cols = frame.shape[1]
            rows = frame.shape[0]

            if cols / float(rows) > WHRatio:
                cropSize = (int(rows * WHRatio), rows)
            else:
                cropSize = (cols, int(cols / WHRatio))

            y1 = int((rows - cropSize[1]) / 2)
            y2 = y1 + cropSize[1]
            x1 = int((cols - cropSize[0]) / 2)
            x2 = x1 + cropSize[0]
            frame = frame[y1:y2, x1:x2]

            cols = frame.shape[1]
            rows = frame.shape[0]

            # Iterate over every detection
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # Once confidence is greater than the threshold we get our bounding box
                if confidence > thr:
                    class_id = int(detections[0, 0, i, 1])

                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop = int(detections[0, 0, i, 5] * cols)
                    yRightTop = int(detections[0, 0, i, 6] * rows)

                    # Draw our bounding box over our image
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0), 3)
                    # Get our class names and put them on our image (using a white background)
                    if class_id in classNames:
                        label = classNames[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                      (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                      (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Convert the result image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the output image using Streamlit
            tab5.image(frame_rgb, caption='Processed Image', use_container_width=True, channels="BGR")
        #
        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_SSDs, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder5.pyplot(fig)
# ______________________________________________________________________________________________
        colorization = tab6.selectbox("Colorizing black and white images by deep learning",["colorization_deploy_v2"])
        colorization_options = tab6.expander("Colorizing black and white images by deep learning")
        placeholder6 = tab6.empty()
        image_colorization = image_rgb.copy()

        if colorization == "colorization_deploy_v2":
            # Set the file path and kernel
            file_path = "./pages/colorize/colorize/blackandwhite/"
            kernel = './pages/colorize/colorize/pts_in_hull.npy'

            # Load the pre-trained model
            net = cv2.dnn.readNetFromCaffe("./pages/colorize/colorize/colorization_deploy_v2.prototxt",
                                           "./pages/colorize/colorize/colorization_release_v2.caffemodel")

            # Load cluster centers
            pts_in_hull = np.load(kernel)

            # Populate cluster centers as 1x1 convolution kernel
            pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
            net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

            # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            # if uploaded_file is not None:
            #     # Convert the uploaded file to an OpenCV image
            #     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            #     # img = cv2.imdecode(file_bytes, 1)

            img = image_colorization


            img_rgb = (img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)

            # Pull out L channel
            img_l = img_lab[:, :, 0]

            # Get original image size
            (H_orig, W_orig) = img_rgb.shape[:2]

            # Resize image to network input size
            img_rs = cv2.resize(img_rgb, (224, 224))

            # Resize image to network input size
            img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
            img_l_rs = img_lab_rs[:, :, 0]

            # Subtract 50 for mean-centering
            img_l_rs -= 50

            net.setInput(cv2.dnn.blobFromImage(img_l_rs))

            # Get the result
            ab_dec = net.forward('class8_ab')[0, :, :, :].transpose((1, 2, 0))

            (H_out, W_out) = ab_dec.shape[:2]
            ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
            img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)

            # Concatenate with original image L
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            # Resize the colorized image to its original dimensions
            img_bgr_out = cv2.resize(img_bgr_out, (W_orig, H_orig), interpolation=cv2.INTER_AREA)

            # Convert the result image to uint8 format for display
            img_bgr_out = (img_bgr_out * 255).astype(np.uint8)

            # Display the images using Streamlit
            tab6.image(img, caption='Original Black and White Image', use_container_width=True)
            tab6.image(img_bgr_out, caption='Colorized Image', use_container_width=True)

        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_colorization, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder6.pyplot(fig)
# ______________________________________________________________________________________________
        Inpainting = tab7.selectbox("Inpainting to Restore Damaged Photos", ["Dilated Mask"])
        Inpainting_options = tab7.expander("Inpainting to Restore Damaged Photos")
        placeholder7 = tab7.empty()
        image_Inpainting = image_rgb.copy()

        if Inpainting == "Dilated Mask":
            # Function to process the images and perform inpainting
            def restore_image(damaged_image, mask_image):
                # Convert the uploaded files to OpenCV images
                damaged_image = cv2.imdecode(np.frombuffer(damaged_image.read(), np.uint8), 1)
                mask_image = cv2.imdecode(np.frombuffer(mask_image.read(), np.uint8), 0)

                # Threshold the mask to make sure it's binary
                ret, thresh1 = cv2.threshold(mask_image, 254, 255, cv2.THRESH_BINARY)

                # Dilate the mask to make the marks thicker
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.dilate(thresh1, kernel, iterations=1)

                # Perform inpainting
                restored = cv2.inpaint(damaged_image, mask, 3, cv2.INPAINT_TELEA)

                return damaged_image, mask, restored


            # Streamlit application
            tab7.title("Photo Restoration using Inpainting")
            tab7.write("Upload a damaged photo and a mask image indicating the damaged areas to restore the photo:")

            uploaded_damaged_file = tab7.file_uploader("Choose a damaged photo...", type=["jpg", "jpeg", "png"])
            uploaded_mask_file = tab7.file_uploader("Choose a mask image...", type=["jpg", "jpeg", "png"])

            if uploaded_damaged_file is not None and uploaded_mask_file is not None:
                damaged_image, mask_image, restored_image = restore_image(uploaded_damaged_file, uploaded_mask_file)

                # Convert images for display
                damaged_image = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB)
                restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)

                # Display the images
                tab7.image(damaged_image, caption='Original Damaged Photo', use_container_width=True)
                tab7.image(mask_image, caption='Mask Image', use_container_width=True, channels='GRAY')
                tab7.image(restored_image, caption='Restored Photo', use_container_width=True)

            # Add a note about the method used
            tab7.write("""
                This app uses OpenCV's inpainting method to restore photos.
                The damaged areas should be marked in white on the mask image.
            """)

# ______________________________________________________________________________________________
        DenoisingColored = tab8.selectbox("Add and Remove Noise and Fix Contrast with Histogram Equalization",["fastNlMeansDenoisingColored","histogram cdf_normalized","histogram equalizeHist","Equlize all RGB"])
        DenoisingColored_options = tab8.expander("Add and Remove Noise and Fix Contrast with Histogram Equalization")
        placeholder8 = tab8.empty()
        image_DenoisingColored = image_rgb.copy()


        def addWhiteNoise(image):
            # Set the range for a random probablity
            # A large prob will mean more noise
            prob = random.uniform(0.05, 0.1)

            # Generate a random matrix in the shape of our input image
            rnd = np.random.rand(image.shape[0], image.shape[1])

            # If the random values in our rnd matrix are less than our random probability
            # We randomly change that pixel in our input image to a value within the range specified
            image[rnd < prob] = np.random.randint(50, 230)
            return image

        if DenoisingColored == "fastNlMeansDenoisingColored":
            # Load our image
            image = image_DenoisingColored
            # imshow("Input Image", image)
            tab8.image(image, caption='Input Image', use_container_width=True)

            # Apply our white noise function to our input image
            noise_1 = addWhiteNoise(image)
            # imshow("Noise Added", noise_1)
            tab8.image(noise_1, caption='Noise Added', use_container_width=True)

            dst = cv2.fastNlMeansDenoisingColored(noise_1, None, 11, 6, 7, 21)

            # imshow("Noise Removed", dst)
            tab8.image(dst, caption='Noise Removed', use_container_width=True)

        if DenoisingColored == "histogram cdf_normalized":
            # Function to process the image and create the histogram and CDF
            def process_image(image):
                # Convert the uploaded file to an OpenCV image
                image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)

                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Create histogram distribution
                hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

                # Get the cumulative sum
                cdf = hist.cumsum()

                # Get a normalized cumulative distribution
                cdf_normalized = cdf * float(hist.max()) / cdf.max()

                return image, gray_image, hist, cdf_normalized


            # Streamlit application
            tab8.title("Histogram and CDF Visualization")
            tab8.write("Upload an image to see its histogram and cumulative distribution function (CDF):")

            uploaded_file = tab8.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                original_image, gray_image, hist, cdf_normalized = process_image(uploaded_file)

                # Display the original and grayscale images
                tab8.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Original Image',
                         use_container_width=True)
                tab8.image(gray_image, caption='Grayscale Image', use_container_width=True, channels='GRAY')

                # Plot the histogram and CDF
                fig, ax = plt.subplots()
                ax.plot(cdf_normalized, color='b')
                ax.hist(gray_image.flatten(), 256, [0, 256], color='r')
                ax.set_xlim([0, 256])
                ax.legend(('cdf', 'histogram'), loc='upper left')

                st.pyplot(fig)

            # Add a note about the method used
            tab8.write("""
                This app calculates the histogram and cumulative distribution function (CDF) of the uploaded image.
                The histogram shows the distribution of pixel intensities, and the CDF shows the cumulative distribution of these intensities.
            """)

        if DenoisingColored == "histogram equalizeHist":
            # Function to process the image and create the histogram and CDF after histogram equalization
            def process_image(image):
                # Convert the uploaded file to an OpenCV image
                image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)

                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Equalize the histogram
                equalized_image = cv2.equalizeHist(gray_image)

                # Create histogram distribution
                hist, bins = np.histogram(equalized_image.flatten(), 256, [0, 256])

                # Get the cumulative sum
                cdf = hist.cumsum()

                # Get a normalized cumulative distribution
                cdf_normalized = cdf * float(hist.max()) / cdf.max()

                return image, equalized_image, hist, cdf_normalized


            # Streamlit application
            tab8.title("Histogram Equalization and CDF Visualization")
            tab8.write(
                "Upload an image to see its histogram equalized version and cumulative distribution function (CDF):")

            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                original_image, equalized_image, hist, cdf_normalized = process_image(uploaded_file)

                # Display the original and equalized images
                tab8.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Original Image',
                         use_container_width=True)
                tab8.image(equalized_image, caption='Equalized Grayscale Image', use_container_width=True, channels='GRAY')

                # Plot the histogram and CDF
                fig, ax = plt.subplots()
                ax.plot(cdf_normalized, color='b')
                ax.hist(equalized_image.flatten(), 256, [0, 256], color='r')
                ax.set_xlim([0, 256])
                ax.legend(('cdf', 'histogram'), loc='upper left')

                tab8.pyplot(fig)

            # Add a note about the method used
            tab8.write("""
                This app performs histogram equalization on the uploaded image and visualizes the histogram and cumulative distribution function (CDF).
                The histogram shows the distribution of pixel intensities, and the CDF shows the cumulative distribution of these intensities.
            """)

        if DenoisingColored == "Equlize all RGB":
            Colored_RGB = tab8.selectbox("Equlize all RGB (BGR) channels of this image and then merge them together to obtain an equlized color image",
                                              ["Original", "Red", "Green","Blue", "Merged"])
            img = image_DenoisingColored
            if Colored_RGB == "Original":
                # imshow("Original", img)
                tab8.image(img, caption='Original', use_container_width=True)
            # Equalize our Histogram
            # Default color format is BGR

            red_channel = img[:, :, 2]
            red = cv2.equalizeHist(red_channel)

            green_channel = img[:, :, 1]
            green = cv2.equalizeHist(green_channel)

            blue_channel = img[:, :, 0]
            blue = cv2.equalizeHist(blue_channel)


            if Colored_RGB == "Red":
                # create empty image with same shape as that of src image
                red_img = np.zeros(img.shape)
                red_img[:, :, 0] = red
                red_img = np.array(red_img, dtype=np.uint8)
                # imshow("Red", red_img)
                tab8.image(red_img, caption='Red', use_container_width=True, channels='RGB')

            if Colored_RGB == "Green":
                green_img = np.zeros(img.shape)
                green_img[:, :, 1] = green
                green_img = np.array(green_img, dtype=np.uint8)
                # imshow("Green", green_img)
                tab8.image(green_img, caption='Green', use_container_width=True, channels='RGB')

            if Colored_RGB == "Blue":
                blue_img = np.zeros(img.shape)
                blue_img[:, :, 2] = blue
                blue_img = np.array(blue_img, dtype=np.uint8)
                # imshow("Blue", blue_img)
                tab8.image(blue_img, caption='Blue', use_container_width=True, channels='RGB')

            if Colored_RGB == "Merged":
                merged = cv2.merge([blue, green, red])
                # imshow("Merged", merged)
                tab8.image(merged, caption='Merged', use_container_width=True, channels='RGB')


        # # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_DenoisingColored, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder8.pyplot(fig)
# ______________________________________________________________________________________________
        Gaussian = tab9.selectbox("Blur Detection - Finding In-focus Images", ["GaussianBlur"])
        Gaussian_options = tab9.expander("Blur Detection - Finding In-focus Images")
        placeholder9 = tab9.empty()
        image_Gaussian = image_rgb.copy()


        # Function to calculate the blur score using the Laplacian variance method
        def get_blur_score(image):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(image, cv2.CV_64F).var()


        # Function to process the image and apply Gaussian blur
        def process_image(image):
            # Convert the uploaded file to an OpenCV image
            # image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
            image = image

            # Apply Gaussian blurs with different kernel sizes
            blur_1 = cv2.GaussianBlur(image, (5, 5), 0)
            blur_2 = cv2.GaussianBlur(image, (9, 9), 0)
            blur_3 = cv2.GaussianBlur(image, (13, 13), 0)

            # Calculate blur scores
            original_score = get_blur_score(image)
            blur_1_score = get_blur_score(blur_1)
            blur_2_score = get_blur_score(blur_2)
            blur_3_score = get_blur_score(blur_3)

            return image, blur_1, blur_2, blur_3, original_score, blur_1_score, blur_2_score, blur_3_score


        if Gaussian == "GaussianBlur":

            original_image, blur_1, blur_2, blur_3, original_score, blur_1_score, blur_2_score, blur_3_score = process_image(
                image_Gaussian)

            # Display the original and blurred images
            tab9.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Original Image', use_container_width=True)
            tab9.write(f"Blur Score (Original Image): {original_score:.2f}")

            tab9.image(cv2.cvtColor(blur_1, cv2.COLOR_BGR2RGB), caption='Blurred Image (5x5)', use_container_width=True)
            tab9.write(f"Blur Score (5x5): {blur_1_score:.2f}")

            tab9.image(cv2.cvtColor(blur_2, cv2.COLOR_BGR2RGB), caption='Blurred Image (9x9)', use_container_width=True)
            tab9.write(f"Blur Score (9x9): {blur_2_score:.2f}")

            tab9.image(cv2.cvtColor(blur_3, cv2.COLOR_BGR2RGB), caption='Blurred Image (13x13)', use_container_width=True)
            tab9.write(f"Blur Score (13x13): {blur_3_score:.2f}")

            # Add a note about the method used
            tab9.write("""
                    This app applies Gaussian blur with different kernel sizes to the uploaded image and calculates the blur score using the Laplacian variance method.
                    The blur score indicates the level of sharpness or blurriness in the image, with a lower score indicating more blur.
                """)

        #     # Example processing: Convert to grayscale
        # image = cv2.cvtColor(image_Gaussian, cv2.COLOR_RGB2BGR)
        # # Plotting with Matplotlib
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # # ax.axis('on')  # Hide axes
        # placeholder9.pyplot(fig)

# ______________________________________________________________________________________________

    except Exception as e:
        st.error(f"Error processing image: {e}")
    else:
        st.sidebar.write("Please upload an image file.")
# ______________________________________________________________________________________________
Recognitions = tab10.selectbox("simple Face Recognitions using the python library face-recognition", ["face_recognition match1","face_recognition match2"])
Recognitions_options = tab10.expander("simple Face Recognitions using the python library face-recognition")
placeholder10 = tab10.empty()
# image_Recognitions = image_rgb.copy()


# Function to load image and encode faces
def load_and_encode_image(image_file):
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    else:
        st.warning("No faces found in the uploaded image.")
        return None

if Recognitions == "face_recognition match1":

    # Streamlit application
    # tab10.title("Face Recognition App")
    tab10.subheader("Upload two images to compare the faces and see if they match:")

    uploaded_file1 = tab10.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"], key="image1")
    uploaded_file2 = tab10.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"], key="image2")

    if uploaded_file1 and uploaded_file2:
        # Load and encode the images
        biden_encoding = load_and_encode_image(uploaded_file1)
        unknown_encoding = load_and_encode_image(uploaded_file2)

        if biden_encoding is not None and unknown_encoding is not None:
            # Compare faces
            result = face_recognition.compare_faces([biden_encoding], unknown_encoding)[0]

            # Display the images and the result
            Recognitions_options.image([uploaded_file1, uploaded_file2], caption=['First Image', 'Second Image'],
                     use_container_width=True)
            tab10.subheader(f"Face Match is {result}")

    # Add a note about the method used
    tab10.write("""
        This app uses face_recognition library to compare faces in two uploaded images.
        The result indicates whether the faces in the two images match or not.
    """)
if Recognitions == "face_recognition match2":

    # Function to load and encode faces from an uploaded file
    def load_and_encode_image(image_file):
        image = face_recognition.load_image_file(image_file)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
        else:
            st.warning("No faces found in the uploaded image.")
            return None


    # Function to process a frame for face recognition
    def recognize_faces(frame, known_face_encodings, known_face_names):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame



    tab10.write("Upload images of known individuals and an image to recognize faces in. Known names: Donald Trump, Joe Biden")


    uploaded_file1 = tab10.file_uploader("Choose an image of the first known individual...",
                                      type=["jpg", "jpeg", "png"], key="known1")
    uploaded_file2 = tab10.file_uploader("Choose an image of the second known individual...",
                                      type=["jpg", "jpeg", "png"], key="known2")
    uploaded_image = tab10.file_uploader("Choose an image to recognize faces in...", type=["jpg", "jpeg", "png"],
                                      key="unknown")

    if uploaded_file1 and uploaded_file2 and uploaded_image:
        # uploaded_file1 = read_and_process_image(uploaded_file1)
        # uploaded_file2 = read_and_process_image(uploaded_file2)

        trump_face_encoding = load_and_encode_image(uploaded_file1)
        biden_face_encoding = load_and_encode_image(uploaded_file2)

        if trump_face_encoding is not None and biden_face_encoding is not None:
            known_face_encodings = [trump_face_encoding, biden_face_encoding]
            known_face_names = ["Donald Trump", "Joe Biden"]

            # uploaded_image = read_and_process_image(uploaded_image)
            unknown_image = face_recognition.load_image_file(uploaded_image)
            frame = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

            result_frame = recognize_faces(frame, known_face_encodings, known_face_names)

            tab10.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), caption='Face Recognition Result',
                     use_container_width=True)

    # Add a note about the method used
    tab10.write("""
        This app uses the face_recognition library to detect and recognize faces in an uploaded image.
        It compares the faces found in the uploaded image with the faces of known individuals uploaded before.
    """)

