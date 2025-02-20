import streamlit as st
import joblib
import numpy as np
from streamlit import slider

knn = joblib.load('knnmodel.pkl')
st.set_page_config(layout="wide")

prediction = None
st.title("Iris Flower Predictor")
with st.container():
    left_column, right_column = st.columns([3,3])
    with left_column:
        st.write("Input Features")
        sepal_length = st.slider("Sepal length (cm)",float(4.0),float(8.0),float(5.8))
        sepal_width = st.slider("Sepal width (cm)",float(1.8),float(4.5),float(3.0))
        petal_length = st.slider("Petal length (cm)",float(1.0),float(7.0),float(3.8))
        petal_width = st.slider("Petal width (cm)", float(0.0),float(2.6),float(1.2))

        features_name = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        # st.button("Classify")
        if st.button("Classify"):
            prediction = knn.predict(features_name)[0]

            iris_species = ["Setosa","Versicolor","Virginica"]
            st.subheader(f"Predicted Iris species is {iris_species[prediction]}")

    with right_column:

        if prediction == 0:
            st.image('Irissetosa1.jpg')
        elif prediction == 1:
            st.image('irisversicolor1.jpg')
        elif prediction ==2:
            st.image('irisverginica1.jpg')


