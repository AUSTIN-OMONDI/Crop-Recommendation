import pickle
from flask import Flask, request, jsonify,url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)
##Load the model
cropmodel=pickle.load(open('crop_model.pkl','rb'))
scalar=pickle.load(open('scaler.pkl','rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=cropmodel.predict(new_data) 
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
# import pickle