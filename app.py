from flask import Flask,request,render_template,app,url_for
import pandas as pd
import numpy as np
import pickle
 
app= Flask(__name__)

model = pickle.load(open('gradient_boosting_model.pkl','rb'))
scale = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    smoker = int(request.form.get("smoker",0))
    data=[age, bmi, smoker]
    print(data)
    final_input = scale.transform(np.array(data).reshape(1,-1))
    output=model.predict(final_input)
    return render_template("index.html",prediction_text="The Insurance Premium Prediction is {}".format(output[0]))

if __name__ == '__main__':
    app.run(debug=True)