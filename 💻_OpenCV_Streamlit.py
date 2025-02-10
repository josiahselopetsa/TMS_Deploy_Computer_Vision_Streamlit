import streamlit as st


st.set_page_config(page_title="Opecv on Streamlit", page_icon="computer",layout="wide")

image = "./image/OpenCV_Chpt_3.png"


with st.container():
    st.subheader("OpenCV part1 on Streamlit examples")
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./image/OpenCV_Chpt_3.png"
    image2 = "./image/OpenCV_Chpt_5.png"
    image3 = "./image/OpenCV_Chpt_7.png"
    image4 = "./image/OpenCV_Chpt_8.png"
    with col1:
        st.image(image1, caption="Manipulate a color space")
    with col2:
        st.image(image2, caption="Perform Image Translations")
    with col3:
        st.image(image3, caption="Bitwise Operations")
    with col4:
        st.image(image4, caption="Convolution Operations, Blurring, Denoising, Sharpening")
st.divider()

with st.container():
    st.subheader("OpenCV part2 on Streamlit examples")
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./image/OpenCV_Chpt_11.png"
    image2 = "./image/OpenCV_Chpt_12.png"
    image3 = "./image/OpenCV_Chpt_16.png"
    image4 = "./image/OpenCV_Chpt_19.png"
    with col1:
        st.image(image1, caption="Contouring Modes (Simple vs Approx)")
    with col2:
        st.image(image2, caption="Sort Contours by Area")
    with col3:
        st.image(image3, caption="Face and Eye Detection with Haar Cascade Classifiers")
    with col4:
        st.image(image4, caption="View the RGB Histogram representations of images")
st.divider()

with st.container():
    st.subheader("OpenCV part3 on Streamlit examples")
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./image/OpenCV_Chpt_21.png"
    image2 = "./image/OpenCV_Chpt_24.png"
    image3 = "./image/OpenCV_Chpt_27.png"
    image4 = "./image/OpenCV_Chpt_30.png"
    with col1:
        st.image(image1, caption=" HSV Color Space to Filter by Color")
    with col2:
        st.image(image2, caption="Mean Shift Algorithm in OpenCV")
    with col3:
        st.image(image3, caption="Apply Facial Landmark Detection")
    with col4:
        st.image(image4, caption="GrabCut Algorithm for background Removal")
st.divider()

with st.container():
    st.subheader("OpenCV part4 on Streamlit examples")
    col1,col2,col3,col4 = st.columns(4)
    image1 = "./image/OpenCV_Chpt_31.png"
    image2 = "./image/OpenCV_Chpt_32.png"
    image3 = "./image/OpenCV_Chpt_35.png"
    image4 = "./image/OpenCV_Chpt_40.png"
    with col1:
        st.image(image1, caption="Optical Character Recognition with PyTesseract & EASY OCR")
    with col2:
        st.image(image2, caption="Barcode Generation and Reading")
    with col3:
        st.image(image3, caption="Pre-trained models to implement an SSD in OpenCV")
    with col4:
        st.image(image4, caption="Simple Face Recognitions using the python library face-recognition")

st.divider()