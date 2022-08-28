import pandas as pd 
from flask import Flask, request,render_template,jsonify
import keras

app = Flask(__name__)

model=keras.models.load_model("Beekeeping.h5")

@app.route('/')
def home():
    return render_template('Beekeeping.html')


@app.route('/predict1',methods=['POST'])
def predict1():
    test_features1=[x for x in request.form.values()]
    test_features1=[eval(i) for i in test_features1]
    final_data=pd.DataFrame([test_features1],columns=['Temperature', 'Humidity'])
    final_data=pd.DataFrame.to_numpy(final_data)
    
    final_data = final_data.reshape((final_data.shape[0], 1, final_data.shape[1]))

    prediction=model.predict(final_data)
    output1=prediction
    
    test_features2=[x for x in request.form.values()]
    test_features2=[eval(i) for i in test_features2]
    final_data=pd.DataFrame([test_features2],columns=['Temperature', 'Humidity'])
    final_data=pd.DataFrame.to_numpy(final_data)
    
    final_data = final_data.reshape((final_data.shape[0], 1, final_data.shape[1]))

    prediction=model.predict(final_data)
    output2=prediction
    
    return render_template('Beekeeping.html', prediction_text='Weight of the box f is approximately {} and Weight of box h is approximately {}'.format(output1,output2))
    

if __name__=="__main__":
    pd.set_option('display.max_columns',None)
    app.run(debug=True)
    
