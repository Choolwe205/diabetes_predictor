import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import matplotlib.pyplot as plt

# Helper function for BMI category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler safely
model_path = os.path.join(script_dir, "model.joblib")
scaler_path = os.path.join(script_dir, "scaler.joblib")

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

languages = ['English', 'Spanish', 'Bahasa Melayu', 'German', 'Italian']
lang_dict = {
    'English': {
        'title': "Diabetes Prediction App",
        'predict': "Predict",
        'download': "Download PDF Report",
        'education': "Education",
        'about': "About",
        'name': "Name",
        'result': "Result",
        'confidence': "Confidence",
        'generated_on': "Generated on"
    },
    'Spanish': {
        'title': "Aplicación de Predicción de Diabetes",
        'predict': "Predecir",
        'download': "Descargar Informe PDF",
        'education': "Educación",
        'about': "Acerca de",
        'name': "Nombre",
        'result': "Resultado",
        'confidence': "Confianza",
        'generated_on': "Generado el"
    },
    'Bahasa Melayu': {
        'title': "Aplikasi Ramalan Diabetes",
        'predict': "Ramalkan",
        'download': "Muat Turun Laporan PDF",
        'education': "Pendidikan",
        'about': "Tentang",
        'name': "Nama",
        'result': "Keputusan",
        'confidence': "Kepastian",
        'generated_on': "Dijana pada"
    },
    'German': {
        'title': "Diabetes Vorhersage App",
        'predict': "Vorhersagen",
        'download': "PDF-Bericht herunterladen",
        'education': "Aufklärung",
        'about': "Über",
        'name': "Name",
        'result': "Ergebnis",
        'confidence': "Sicherheit",
        'generated_on': "Erstellt am"
    },
    'Italian': {
        'title': "App Predizione Diabete",
        'predict': "Prevedi",
        'download': "Scarica Report PDF",
        'education': "Educazione",
        'about': "Informazioni",
        'name': "Nome",
        'result': "Risultato",
        'confidence': "Affidabilità",
        'generated_on': "Generato il"
    }
}


st.sidebar.title("Navigation")
lang = st.sidebar.selectbox("Language", languages)
page = st.sidebar.radio("Pages", ["Home", "Prediction", "Download Report", "Education", "About"])
text = lang_dict[lang]

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name.strip())

def create_pdf(report, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Diabetes Risk Report")
    c.setFont("Helvetica", 12)
    y = height - 90

    c.drawString(50, y, f"{text['generated_on']}: {report['Generated']}")
    y -= 25
    c.drawString(50, y, f"{text['name']}: {report['Name']}")
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Input Parameters:")
    y -= 25
    c.setFont("Helvetica", 12)
    for key, val in report['Inputs'].items():
        c.drawString(70, y, f"{key}: {val}")
        y -= 20
    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"{text['result']}: {report['Result']}")
    y -= 25
    c.drawString(50, y, f"{text['confidence']}: {report['Confidence']}")
    c.save()
if page == "Home":
    st.title(f" {text['title']} ")

    # Intro text with markdown styling
    st.markdown("""
    ### Welcome to the Diabetes Prediction App!
    
    This app helps you **assess your risk** of diabetes using simple health metrics.
    
    - Easy to use sliders and inputs  
    - Get instant predictions with confidence scores  
    - Download detailed reports for your records  
    - Learn how to manage and prevent diabetes  
    
    Let's take a step towards a healthier you! 
    """)
    
    # Motivational Quote in italics
    st.markdown("> _“The greatest wealth is health.”_ Virgil")
    
    # Columns for quick facts
    col1, col2, col3 = st.columns(3)
    col1.metric("Global Diabetes Cases", "537M", delta="+16M YoY")
    col2.metric("Annual Deaths", "6.7M", delta="+100K YoY")
    col3.metric("Prevention", "80% Cases Preventable")


elif page == "Prediction":
    st.title(text['predict'])

    name = st.text_input(text['name'], value=st.session_state.get('name', ''))
    
    glucose = st.slider(
        'Glucose (mg/dL)', 50, 300, st.session_state.get('glucose', 120),
        help="Measure of blood sugar level. Normal fasting glucose is 70-140 mg/dL."
    )
    systolic = st.slider(
        'Systolic BP (mmHg)', 70, 200, st.session_state.get('systolic', 120),
        help="The higher number in blood pressure; pressure when heart beats."
    )
    diastolic = st.slider(
        'Diastolic BP (mmHg)', 40, 130, st.session_state.get('diastolic', 80),
        help="The lower number in blood pressure; pressure when heart rests."
    )
    height_cm = st.slider(
        'Height (cm)', 120, 210, st.session_state.get('height_cm', 170),
        help="Your height in centimeters."
    )
    weight_kg = st.slider(
        'Weight (kg)', 30, 200, st.session_state.get('weight_kg', 70),
        help="Your weight in kilograms."
    )
    age = st.slider(
        'Age', 18, 100, st.session_state.get('age', 40),
        help="Your age in years."
    )

    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_cat = bmi_category(bmi)
    st.write(f"BMI: {bmi:.2f} ({bmi_cat})")

    if glucose < 70:
        st.warning("Glucose below normal fasting levels. Please verify your input.")
    elif glucose > 200:
        st.warning("High glucose level detected. Please consult a doctor.")

    if st.button(text['predict']):
        if not name.strip():
            st.warning("Please enter your name before predicting.")
        else:
            st.session_state.update({
                'name': name,
                'glucose': glucose,
                'systolic': systolic,
                'diastolic': diastolic,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'age': age
            })

            input_df = pd.DataFrame(
                [[glucose, systolic, diastolic, height_cm, weight_kg, age]],
                columns=['Glucose', 'Systolic', 'Diastolic', 'Height_cm', 'Weight_kg', 'Age']
            )
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            result = "Likely Diabetic" if pred == 1 else "Unlikely Diabetic"

            st.session_state['report'] = {
                'Name': name,
                'Inputs': {
                    'Glucose': glucose,
                    'Systolic': systolic,
                    'Diastolic': diastolic,
                    'Height_cm': height_cm,
                    'Weight_kg': weight_kg,
                    'Age': age,
                    'BMI': f"{bmi:.2f} ({bmi_cat})"
                },
                'Result': result,
                'Confidence': f"{prob:.2%}",
                'Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            st.success(f"{text['result']}: {result}")
            st.progress(int(prob * 100))
            st.info(f"{text['confidence']}: {prob:.2%}")

            # Feature importance visualization
            st.subheader("Feature Importance")
            try:
                # If model has feature_importances_ attribute (e.g. tree-based)
                importances = model.feature_importances_
                features = ['Glucose', 'Systolic', 'Diastolic', 'Height_cm', 'Weight_kg', 'Age']
            except AttributeError:
                # Otherwise try coef_ (e.g. linear models)
                importances = model.coef_[0]
                features = ['Glucose', 'Systolic', 'Diastolic', 'Height_cm', 'Weight_kg', 'Age']
            
            # Make importances absolute and normalize
            importances = np.abs(importances)
            importances = importances / importances.sum()
            
            fig, ax = plt.subplots()
            ax.barh(features, importances, color='skyblue')
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance in Model")
            st.pyplot(fig)

elif page == "Download Report":
    st.title(text['download'])
    if 'report' in st.session_state:
        report = st.session_state['report']
        safe_name = sanitize_filename(report['Name']) or "report"
        pdf_filename = f"{safe_name}_diabetes_report.pdf"
        pdf_path = os.path.join(script_dir, pdf_filename)

        create_pdf(report, pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )
    else:
        st.info("No prediction yet.")

elif page == "Education":
    st.title(text['education'])
    st.header("About Diabetes")
    st.write("Diabetes is a long-term condition where blood sugar levels are too high. There are two main types: Type 1 (body does not produce insulin) and Type 2 (body does not use insulin properly). Managing diet, exercise, and regular check-ups help prevent complications.")

    st.header("Healthy Ranges")
    st.write("- Glucose: 70–140 mg/dL (fasting and post-meal).\n- Blood Pressure: less than 120/80 mmHg.\n- BMI: 18.5–24.9 (normal weight).")

    st.header("Prevention Tips")
    st.write("- Maintain a balanced diet with low sugar intake.\n- Exercise regularly.\n- Get regular check-ups and monitor your blood sugar levels.\n- Avoid smoking and excessive alcohol.")

    st.header("Resources")
    st.write("For more information, visit trusted health websites like the WHO, CDC, or consult your healthcare provider.")

    st.header("FAQ")
    with st.expander("What causes diabetes?"):
        st.write("Diabetes is caused by either the body's inability to produce insulin (Type 1) or use insulin effectively (Type 2).")

    with st.expander("How can I prevent diabetes?"):
        st.write("Maintain a healthy diet, exercise regularly, monitor blood sugar levels, and avoid smoking.")

    with st.expander("What are the symptoms?"):
        st.write("Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.")

    with st.expander("When should I see a doctor?"):
        st.write("If you experience symptoms or have risk factors such as obesity, family history, or high blood sugar, consult a healthcare professional.")

elif page == "About":
    st.title(text['about'])
    st.write("Was Intended as a class project. Always consult professionals.")
