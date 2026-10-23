import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the likelihood"
    "of the customer leaving the service."
)

st.info(
    "This dashboard uses a trained Gradient Boosting Model"
    "through the project's FastAPI backend."
)

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        customer_id = st.text_input(
            "Customer ID", value="TEST-1002"
        )

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior_citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.number_input(
            "Tenure (month)", min_value=0, max_value=1000,
            value=12, step=1,
        )

        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )

        streamimg_tv = st.selectbox(
            "Streamimg TV", ["No", "Yes", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed  check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0,
        )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True,
        )


    if submitted:
        if not customer_id.strip():
            st.error("Please enter a customer ID.")
        else:
            payload = {
                "customerID": customer_id.strip(),
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streamimg_tv,
                "StreamingMovies": streamimg_tv,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            try:
                with st.spinner("Analyzing customer information..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=30,
                    )

                if response.status_code == 200:
                    result = response.json()
                    probability = result["churn_probability_percent"]
                    prediction = result["prediction"]

                    st.divider()
                    st.subheader("Prediction Results")

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Customer ID", result["customerID"]
                    )

                    metric2.metric(
                        "Churn Probability", f"{probability:.2f}%"
                    )

                    metric3.metric(
                        "Decision Threshold",
                        f'{result["threshold"]:.2f}',
                    )

                    if prediction == "Churn":
                        st.error(
                            "Prediction: Customer is likely to churn"
                        )
                    else:
                        st.success(
                            "Prediction: Customer is likely to stay"
                        )

                    st.progress(
                        max(0.0, min(1.0, probability / 100))
                    )

                    st.caption(
                        "This is a model estimate, not a guarantee "
                        "of customer behavior. The selected threshold"
                        "is 0.34."
                    )

                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the prediction API."
                    "Make sure the FastAPI server is running at"
                    "http://127.0.0.1:8000."
                )

            except requests.exceptions.Timeout:
                st.error("The API request timed out. Please try again.")

            except (KeyError, ValueError) as error:
                st.error(f"Could not process the API response: {error}")