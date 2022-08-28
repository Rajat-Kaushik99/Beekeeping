import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request,render_template,jsonify
import tensorflow as tf
import keras

app = Flask(__name__)

model=tf.keras.models.load_model("Beekeeping.h5")

@app.route('/')
def home():
    return render_template('Beekeeping.html')


@app.route('/predict',methods=['POST'])
def predict():
    test_features=[x for x in request.form.values()]
    final_data=pd.DataFrame([test_features],columns=['Temperature', 'Humidity'])

    #final=np.array(final_data)
    prediction=model.predict(final_data)
    output=prediction
        
    return render_template('Beekeeping.html', prediction_text='Weight of the box is approximately {}'.format(output))
    

if __name__=="__main__":
    pd.set_option('display.max_columns',None)
    app.run(debug=True)
    
