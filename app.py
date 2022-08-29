import pandas as pd 
from flask import Flask, request,render_template,jsonify
import keras
import json
import plotly
import plotly.express as px
pd.set_option('display.max_rows', 500)
beehive_df = pd.read_csv('beehive_humidity_temp_weight.csv')

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
    
    
    return render_template('Beekeeping.html', prediction_text='Weight of the box f is approximately {}'.format(output))
    
@app.route('/chart1')
def chart1():
    
    fig = px.line(beehive_df, x='timestamp', y="humidity")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  
    return render_template('notdash2.html', graphJSON=graphJSON)

@app.route('/chart2')
def chart2():

    fig = px.line(beehive_df, x='timestamp', y="weight")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('notdash2.html', graphJSON=graphJSON)

@app.route('/chart3')
def chart3():

    fig = px.line(beehive_df, x='timestamp', y="temperature")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('notdash2.html', graphJSON=graphJSON)

@app.route('/chart4')
def chart4():

    fig = px.scatter(beehive_df, x="temperature", y="weight", trendline="ols")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('notdash2.html', graphJSON=graphJSON)

@app.route('/chart5')
def chart5():

    fig = px.scatter(beehive_df, x="humidity", y="weight", trendline="ols")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('notdash2.html', graphJSON=graphJSON)


if __name__=="__main__":
    pd.set_option('display.max_columns',None)
    app.run(debug=True)
    
