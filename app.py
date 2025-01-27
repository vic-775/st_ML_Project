import streamlit as st
import joblib
import pandas as pd

# Load the trained model, scaler, and the training columns
model = joblib.load('models/nn_classifier.jb')  # Update the path if necessary
scaler = joblib.load('models/minmax_scaler.pkl')  # Update the path if necessary
x_train = pd.read_csv('models/x_dum.csv')  # Update the path if necessary

# Streamlit app title
st.title('Loan Approval Prediction')

# Initialize session state for clearing inputs
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

# Create a form to collect user input
with st.form(key='loan_form'):
    name = st.text_input('Name', value="" if st.session_state.form_submitted else None)
    dependants = st.number_input('Dependants', min_value=0, value=0 if st.session_state.form_submitted else None)
    applicants_income = st.number_input('Applicant Income', min_value=0, value=0 if st.session_state.form_submitted else None)
    coapplicants_income = st.number_input('Coapplicant Income', min_value=0, value=0 if st.session_state.form_submitted else None)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=0 if st.session_state.form_submitted else None)
    loan_term = st.number_input('Loan Term', min_value=0, value=0 if st.session_state.form_submitted else None)
    credit_history = st.number_input('Credit History (0 or 1)', min_value=0, max_value=1, value=0 if st.session_state.form_submitted else None)
    
    # Selectbox with "Choose option" placeholder and no default selection
    gender = st.selectbox('Gender', ['Choose option', 'Female', 'Male'], index=0 if st.session_state.form_submitted else None)
    marital_status = st.selectbox('Marital Status', ['Choose option', 'Yes', 'No'], index=0 if st.session_state.form_submitted else None)
    education_status = st.selectbox('Education Status', ['Choose option', 'Graduated', 'Non_Graduate'], index=0 if st.session_state.form_submitted else None)
    self_employed = st.selectbox('Self Employed', ['Choose option', 'Yes', 'No'], index=0 if st.session_state.form_submitted else None)
    area = st.selectbox('Property Area', ['Choose option', 'Rural', 'Semi-Urban', 'Urban'], index=0 if st.session_state.form_submitted else None)

    # Submit button for the form
    submit_button = st.form_submit_button(label='Submit')

# Execute prediction only if the form is submitted
if submit_button:
    # Reset form submission state
    st.session_state.form_submitted = True

    # Mapping inputs to numerical values
    gender_val = 0 if gender == 'Female' else 1 if gender == 'Male' else None
    marital_status_val = 1 if marital_status == 'Yes' else 0 if marital_status == 'No' else None
    education_status_val = 0 if education_status == 'Graduated' else 1 if education_status == 'Non_Graduate' else None
    self_employed_val = 1 if self_employed == 'Yes' else 0 if self_employed == 'No' else None
    area_val = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}.get(area, None)

    # Check if any mandatory fields are not selected
    if None in [gender_val, marital_status_val, education_status_val, self_employed_val, area_val]:
        st.error("Please fill in all the fields!")
    else:
        # Construct input dictionary
        input_dict = {
            'Dependents': dependants,
            'ApplicantIncome': applicants_income,
            'CoapplicantIncome': coapplicants_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Gender_Female': int(gender_val == 0),
            'Gender_Male': int(gender_val == 1),
            'Married_No': int(marital_status_val == 0),
            'Married_Yes': int(marital_status_val == 1),
            'Education_Graduate': int(education_status_val == 0),
            'Education_Not Graduate': int(education_status_val == 1),
            'Self_Employed_No': int(self_employed_val == 0),
            'Self_Employed_Yes': int(self_employed_val == 1),
            'Property_Area_Rural': int(area_val == 0),
            'Property_Area_Semiurban': int(area_val == 1),
            'Property_Area_Urban': int(area_val == 2),
        }

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Reindex to match the training columns
        input_df = input_df.reindex(columns=x_train.columns, fill_value=0)

        # Scale the input using the scaler loaded
        input_scaled = scaler.transform(input_df)

        # Make the prediction
        prediction = model.predict(input_scaled)[0]  # Use 'input_scaled' for prediction

        # Show the result
        if prediction > 0.5:
            st.write(f"Loan Status: **Loan Approved**")
        else:
            st.write(f"Loan Status: **Loan Not Approved**")

        # Reset form submission state after displaying the result
        st.session_state.form_submitted = False
