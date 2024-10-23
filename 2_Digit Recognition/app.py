import numpy as np
import pandas as pd
import streamlit as st 
# import numpy as np
from tensorflow.keras.models import model_from_json


# opening and store file in a variable
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
# use Keras model_from_json to make a loaded model
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.weights.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


from sqlalchemy import create_engine
def predict(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    
    data_pred = data.values.astype('float32')/255
    data_pred = data_pred.reshape((data_pred.shape[0], 28, 28, 1))  # Reshape to (num_samples, 28, 28, 1)

       
    prediction = pd.DataFrame(np.argmax(loaded_model.predict(data_pred), axis= 1), columns = ['Label'])
    
    final = pd.concat([prediction, data], axis = 1)
    
    final.to_sql('digit_recongnition_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final




def main():
    

    st.title("Digit Recognition")
    st.sidebar.title("Hand Written Digit Recognition")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Neural_Network </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
        
        
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm))
        
                           
if __name__=='__main__':
    main()
            

