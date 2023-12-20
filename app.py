# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO
import psycopg2
from datetime import datetime
from sqlalchemy import create_engine

user = 'root'
pw = 'raviravi12'
db = 'pipe_inventory'

# Replace the relative path to your weight file
model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a connection to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost", 
    database="Pipe Inventory",
    user="postgres",
    password="12345"
)

# Function to insert detection results into the database
def insert_detection_result(timestamp, num_objects):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO detection_results (timestamp, num_objects) VALUES (%s, %s)", (timestamp, num_objects))
    conn.commit()
    cursor.close()

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Pipe Detection and Counting 🤖 ")
st.caption('Innodatatics')
st.caption('Click the Detect Objects button and check the result.')

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if an image is uploaded
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image", use_column_width=True)

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        try:
            with st.expander("Detection Results"):
                num_objects = len(boxes)
                st.write(f"Number of Pipes detected: {num_objects}")
                for box in boxes:
                    st.write(box.xywh)
                # Insert detection result into the PostgreSQL database
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                insert_detection_result(timestamp, num_objects)
        except Exception as ex:
            st.write("No image is uploaded yet!")

# Close the database connection when done
conn.close()
