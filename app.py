
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

model = joblib.load("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction System")
st.write("Enter student details to predict the final mathematics grade.")

with st.form("student_form"):
    st.subheader("Student Information")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.number_input("Age", 15, 22, 17)
    address = st.selectbox("Address", ["U", "R"])
    famsize = st.selectbox("Family Size", ["GT3", "LE3"])
    Pstatus = st.selectbox("Parents' Status", ["T", "A"])

    Medu = st.selectbox("Mother's Education (0–4)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Father's Education (0–4)", [0, 1, 2, 3, 4])

    Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("Reason for Choosing School", ["course", "home", "reputation", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    traveltime = st.selectbox("Travel Time (1–4)", [1, 2, 3, 4])
    studytime = st.selectbox("Weekly Study Time (1–4)", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

    schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
    famsup = st.selectbox("Family Educational Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

    famrel = st.selectbox("Family Relationship Quality (1–5)", [1, 2, 3, 4, 5])
    freetime = st.selectbox("Free Time (1–5)", [1, 2, 3, 4, 5])
    goout = st.selectbox("Going Out (1–5)", [1, 2, 3, 4, 5])
    Dalc = st.selectbox("Workday Alcohol Consumption (1–5)", [1, 2, 3, 4, 5])
    Walc = st.selectbox("Weekend Alcohol Consumption (1–5)", [1, 2, 3, 4, 5])
    health = st.selectbox("Health Status (1–5)", [1, 2, 3, 4, 5])
    absences = st.number_input("School Absences", min_value=0, value=0)

    G1 = st.number_input("First Period Grade (0–20)", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Second Period Grade (0–20)", min_value=0, max_value=20, value=10)

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    input_data = pd.DataFrame([{
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.success(f"Predicted Final Grade (G3): {prediction:.2f} / 20")
    st.caption("This is an educational model estimate, not a guaranteed final grade.")
