import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import tree
import streamlit as st 

from web_function import train_model

def app(df, x, y):
    
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.title("Visualisasi Prediksi Asuransi perjalanan")
    
    if st.checkbox("Plot Confussion Matrix"):
        model, score = train_model(x,y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap='Blues', ax=ax)
        st.pyplot(fig)
        
    if st.checkbox('Plot Decission Tree'):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree= model,
            max_depth=4,
            out_file=None,
            filled=True, 
            rounded=True,
            feature_names=x.columns, 
            class_names=['No','Yes'],
        )
        
        st.graphviz_chart(dot_data,)