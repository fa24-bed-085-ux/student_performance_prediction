from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "student_performance_model.pkl is missing. "
            "Keep it in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


st.title("🎓 Student Performance Prediction System")
st.write("Enter student details to estimate the final Mathematics grade (G3), from 0 to 20.")

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "This model was saved with scikit-learn 1.6.1. "
        "Keep scikit-learn==1.6.1 in requirements.txt and deploy with Python 3.11."
    )
    st.stop()

with st.form("student_form"):
    st.subheader("Student Information")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.number_input("Age", min_value=15, max_value=22, value=17, step=1)
    address = st.selectbox("Address", ["U", "R"])
    famsize = st.selectbox("Family size", ["GT3", "LE3"])
    Pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"])

    Medu = st.selectbox("Mother's education (0–4)", [0, 1, 2, 3, 4], index=2)
    Fedu = st.selectbox("Father's education (0–4)", [0, 1, 2, 3, 4], index=2)
    jobs = ["teacher", "health", "services", "at_home", "other"]
    Mjob = st.selectbox("Mother's job", jobs)
    Fjob = st.selectbox("Father's job", jobs)
    reason = st.selectbox("Reason for choosing school", ["course", "home", "reputation", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    traveltime = st.selectbox("Travel time (1–4)", [1, 2, 3, 4], index=0)
    studytime = st.selectbox("Weekly study time (1–4)", [1, 2, 3, 4], index=1)
    failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)

    yes_no = ["yes", "no"]
    schoolsup = st.selectbox("Extra educational support", yes_no, index=1)
    famsup = st.selectbox("Family educational support", yes_no, index=0)
    paid = st.selectbox("Extra paid classes", yes_no, index=1)
    activities = st.selectbox("Extracurricular activities", yes_no, index=0)
    nursery = st.selectbox("Attended nursery school", yes_no, index=0)
    higher = st.selectbox("Wants higher education", yes_no, index=0)
    internet = st.selectbox("Internet access", yes_no, index=0)
    romantic = st.selectbox("In a romantic relationship", yes_no, index=1)

    famrel = st.selectbox("Family relationship quality (1–5)", [1, 2, 3, 4, 5], index=3)
    freetime = st.selectbox("Free time after school (1–5)", [1, 2, 3, 4, 5], index=2)
    goout = st.selectbox("Going out with friends (1–5)", [1, 2, 3, 4, 5], index=2)
    Dalc = st.selectbox("Workday alcohol consumption (1–5)", [1, 2, 3, 4, 5], index=0)
    Walc = st.selectbox("Weekend alcohol consumption (1–5)", [1, 2, 3, 4, 5], index=1)
    health = st.selectbox("Health status (1–5)", [1, 2, 3, 4, 5], index=2)

    absences = st.number_input("School absences", min_value=0, max_value=500, value=0, step=1)
    G1 = st.number_input("First period grade (G1)", min_value=0, max_value=20, value=10, step=1)
    G2 = st.number_input("Second period grade (G2)", min_value=0, max_value=20, value=10, step=1)

    submitted = st.form_submit_button("Predict final grade")

if submitted:
    # These 32 columns match the features used to train the supplied model.
    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": int(Medu),
        "Fedu": int(Fedu),
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(traveltime),
        "studytime": int(studytime),
        "failures": int(failures),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": int(famrel),
        "freetime": int(freetime),
        "goout": int(goout),
        "Dalc": int(Dalc),
        "Walc": int(Walc),
        "health": int(health),
        "absences": int(absences),
        "G1": int(G1),
        "G2": int(G2),
    }])

    try:
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            input_data = input_data.loc[:, list(expected)]
        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, min(20.0, prediction))
        st.success(f"Predicted final grade (G3): {prediction:.2f} / 20")
        if prediction >= 10:
            st.write("The predicted grade is at or above 10.")
        else:
            st.warning("The predicted grade is below 10.")
        st.caption("This is an estimate from the trained model, not a guaranteed grade.")
    except Exception as exc:
        st.error("Prediction failed. Check that the model and feature columns match.")
        st.code(f"{type(exc).__name__}: {exc}")
