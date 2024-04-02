import cv2
!pip install dlib
!pip install face_recognition
import numpy as np
import face_recognition
import streamlit as st

st.title("Face Recognition")
arr = []
# Function to find encodings for uploaded images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Upload images from Streamlit sidebar and store them
uploaded_files = st.sidebar.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if uploaded_files:
    images = []
    file_names = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        images.append(img)
        file_names.append(uploaded_file.name)  # Store the name of the uploaded file

    # Find encodings for uploaded images
    encodeList = findEncodings(images)

    # Face recognition
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()
    stop_button = st.sidebar.button("Stop Camera", key='stop_button', help="Click to stop the camera")
    st.markdown(
        """
        <style>
            #stop_button { background-color: red; }
        </style>
        """,
        unsafe_allow_html=True
    )
    while True:
        if stop_button:
            break
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeList, encodeFace)
            faceDis = face_recognition.face_distance(encodeList, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = file_names[matchIndex]  # Get the name of the matching uploaded file
                
            else:
                name = 'Unknown'
            print(name)
            arr.append(name)
            print(arr)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
        # Update the displayed image with the latest frame
        image_placeholder.image(img, channels="BGR", use_column_width=True, caption="Live Video Stream")

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    st.write("present students are:")
    print(arr)
    st.write(arr)
