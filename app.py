import pandas as pd 
from flask import Flask, request,render_template,jsonify
import keras

app = Flask(__name__)

model=keras.models.load_model("Beekeeping.h5")

@app.route('/')
def home():
    return render_template('Beekeeping.html')


@app.route('/predict',methods=['POST'])
def predict():
    test_features=[x for x in request.form.values()]
    test_features=[eval(i) for i in test_features]
    final_data=pd.DataFrame([test_features],columns=['Temperature', 'Humidity'])
    final_data=pd.DataFrame.to_numpy(final_data)
    
    final_data = final_data.reshape((final_data.shape[0], 1, final_data.shape[1]))

    prediction=model.predict(final_data)
    output=prediction
    return render_template('Beekeeping.html', prediction_text='Weight of the box is approximately {}'.format(output))
    

if __name__=="__main__":
    pd.set_option('display.max_columns',None)
    app.run(debug=True)
    
