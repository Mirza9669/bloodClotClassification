import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from fpdf import FPDF
import pandas as pd
import tempfile
import os
import pyrebase


# Firebase Configuration
firebaseConfig = {
  'apiKey': "AIzaSyDJq_rubL9qIqDnfgJC8UAgvxl9TPxKuEw",
  'authDomain': "test-firestore-streamlit-e6b60.firebaseapp.com",
  'projectId': "test-firestore-streamlit-e6b60",
  'databaseURL': "https://test-firestore-streamlit-e6b60-default-rtdb.europe-west1.firebasedatabase.app/",
  'storageBucket': "test-firestore-streamlit-e6b60.appspot.com",
  'messagingSenderId': "270143813073",
  'appId': "1:270143813073:web:df3b34b327351a22681cf0",
  'measurementId': "G-28RP94JZG4"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# Load the pre-trained model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
num_classes = 2  # Replace with the actual number of classes in your model
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('modelcnn.pth'))
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to classify the image
def classify_image(image):
    image = preprocess(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item(), generate_report(image, predicted.item(), image_path)

# Function to generate the PDF report
def generate_report(image, class_label, image_path):
    # Load the CSV file
    df = pd.read_csv('F:\\DATASET\\model(143x143).csv')

    # Create a PDF object
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font and size for the report title
    pdf.set_font("Arial", size=16)

    # Write the report title
    pdf.cell(0, 5, "Report for Blood Clot Image", ln=True, align="C")

    # Set font and size for the report content
    pdf.set_font("Arial", size=12)

    # Write the image information
    pdf.cell(0, 15, "", ln=True)  # Add a blank line
    pdf.image(image_path, x=pdf.w / 3, y=None, w=pdf.w / 3, h=pdf.w / 3)
    pdf.cell(0, 10, "", ln=True)  # Add a blank line
    
    pdf.cell(0, 10, f"Class: {class_label}", ln=True, align="L")
    pdf.cell(0, 10, "", ln=True)  # Add a blank line    

    # Save the PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        pdf.output(temp_filename)

    with open(temp_filename, "rb") as file:
        pdf_contents = file.read()

    return pdf_contents


st.sidebar.title("Stroke Detection")

# Navigation
pages = st.sidebar.radio('Pages', ['Login', 'Sign up'])
upload_folder = "F:\\result"

if pages == 'Login':
    # Obtain User Input for email and password
    email = st.sidebar.text_input('Please enter your email address')
    password = st.sidebar.text_input('Please enter your password', type='password')

    # Login Block
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        bio = st.radio('Jump to', ['Home', 'About', 'Symptoms and Causes', 'Logout'])


    # HOME PAGE
        if bio == 'Home':
            col1, col2 = st.columns(2)

            # col for Profile picture
            with col1:
                uploaded_file = st.file_uploader("Choose a PNG file", type="png")
                if uploaded_file is not None:
                    image_path = os.path.join(upload_folder, uploaded_file.name)
                    
                    image = Image.open(uploaded_file)
                    grayscale_image = image.convert('L')
                    st.image(image, use_column_width=True)
                    grayscale_image.save(image_path)

                    # Classify the image and generate the report
                    prediction, report_path = classify_image(image)

                    # Display the predicted class or any other relevant information
                    if prediction == 0:
                        class_label = 'CE'
                    elif prediction == 1:
                        class_label = 'LAA'    
                                
                    st.write("Predicted Class:", prediction)
                    st.write("Class:", class_label)
                    # Generate the report
                    report_pdf = generate_report(image, class_label, image_path)

                    # Display the PDF report
                    # st.write("Preview Report:")
                    # st.write(report_pdf, format="pdf")

                    # Add a download button for the report
                    st.download_button("Download Report", report_pdf, file_name="report.pdf")

        if bio == 'About':
            st.title('About Blood Clots')
            st.write('Blood clots are gel-like clumps of blood. They are beneficial when they form in response to an injury or a cut, plugging damaged blood vessels, which stops bleeding. But they can also form when they aren\'t needed and cause a heart attack, stroke, or other serious medical problems.')
            st.write('This tool allows you to upload a PNG image of a blood clot for analysis.')

        elif bio == 'Symptoms and Causes':
            st.title('Symptoms and Causes of Blood Clots')
            st.write('Symptoms of a blood clot include warmth over the affected area, pain or tenderness, swelling, and redness or discoloration.')
            st.write('Causes of a blood clot include heart arrhythmias, deep vein thrombosis (DVT), pregnancy, obesity, prolonged sitting or bed rest, smoking, and certain medications.')

        # Logout
        if bio == 'Logout':
            auth.current_user = None  # Clear the current user
            st.caching.clear_cache()  # Clear the cache to refresh the page
            st.experimental_rerun()  # Rerun the app from the beginning

elif pages == 'Sign up':
   # Sign up Block
    name = st.sidebar.text_input('Please input your Name')
    email = st.sidebar.text_input('Please input your Email')
    password = st.sidebar.text_input('Please input your Password', type='password')
    submit = st.sidebar.button('Create my account')
    
    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created successfully!')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(name)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Welcome ' + name)
        st.info('Login via the login drop-down selection')
