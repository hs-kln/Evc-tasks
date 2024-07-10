import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Load the data
df = pd.read_csv(r"C:\Users\hassa\OneDrive\Desktop\stream ved\insurance.csv")  # Update this path

# Encoding
LE = LabelEncoder()
features = ['sex', 'smoker']
for f in features:
    df[f] = LE.fit_transform(df[f])
df = df.drop('region', axis=1)

# Split data into features and target
Y = df.charges
X = df.drop('charges', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=44, shuffle=True)

# Apply Random Forest Regressor Model
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=33)
RandomForestRegressorModel.fit(X_train, y_train)

# Streamlit interface
st.title("Insurance Charges Prediction")

age = st.slider("Age", min_value=18, max_value=100, value=30)
sex = st.radio("Sex", options=["female", "male"])
bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.slider("Children", min_value=0, max_value=10, value=1)
smoker = st.radio("Smoker", options=["no", "yes"])

# Prediction function
def predict_charges(age, sex, bmi, children, smoker):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [0 if sex == "female" else 1],  # Encoding: 0 for female, 1 for male
        'bmi': [bmi],
        'children': [children],
        'smoker': [0 if smoker == "no" else 1],  # Encoding: 0 for no, 1 for yes
    })
    prediction = RandomForestRegressorModel.predict(input_data)[0]
    return prediction

if st.button("Predict"):
    result = predict_charges(age, sex, bmi, children, smoker)
    st.success(f"The predicted insurance charge is: ${result:.2f}")
