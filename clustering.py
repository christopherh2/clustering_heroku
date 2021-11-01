import streamlit as st
import pandas as pd
from pycaret.clustering import *

# Loading the model
model= load_model('IrisKmeans')
description = {}
description['Cluster 0']='This cluster has flowers with long and wide sepals & long and wide petals'
description['Cluster 1']='This cluster has flowers with short and wide sepals & short and narrow petals'
description['Cluster 2']='This cluster has flowers with medium-long and medium-wide sepals & medium-long and medium-wide petals'
description['Cluster 3']='This cluster has flowers with medium-long and narrow sepals & medium-long and medium-wide petals'

# Main function of web app
def run():
    # Sidebar info
    st.sidebar.info('This app is created to analyze Iris flowers')
    
    # Choose type of prediction
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    
    # Add an image
    st.sidebar.image('iris.jpg')
    
    # Single predictions
    if add_selectbox == 'Online':
        dic1={}
        # Numeric input 
        features =['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
        
        st.text("Enter flower's information")
        sepalL = st.number_input('Sepal length (cm)', min_value=4.0, max_value=8.0,value=4.0)
        sepalW = st.number_input('Sepal length (cm)', min_value=2.0, max_value=8.0,value=2.0)
        petalL = st.number_input('Petal length (cm)', min_value=1.0, max_value=7.0,value=1.0)
        petalW = st.number_input('Petal width (cm)', min_value=0.0, max_value=2.7,value=1.0)
        
        
        # Dropdown menu
        species = st.selectbox('Species', ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        
        inputs=[sepalL,sepalW,petalL,petalW,species]
        
        for i in range(5):
            dic1[features[i]]=inputs[i]
        
        # Create dataframe for prediction
        input_df = pd.DataFrame([dic1])
    
        # Make a single predictions    
        if st.button("Predict"):
            prediction = predict_model(model=model,data=input_df)
            cluster = prediction.Cluster[0]
            st.success('This flower belongs to {}'.format(cluster))
            st.success(description[cluster])
        
        
        # File upload
    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        # Perform predictions if there is a file
        if file_upload is not None:
            data = pd.read_csv(file_upload) 
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
        
        
# Run main program if app.py is run directly
if __name__ == '__main__':
    run()
