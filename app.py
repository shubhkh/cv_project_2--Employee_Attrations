import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@st.cache
def load_data():
    df = pd.read_csv('employee_data.csv')  # Replace with your file path
    return df

def preprocess_data(df):
    le = LabelEncoder()
    df['Department'] = le.fit_transform(df['Department'])
    df['salary'] = le.fit_transform(df['salary'])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('left', axis=1))
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    df_scaled['left'] = df['left']
    return df_scaled
