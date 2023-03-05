import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, json, render_template, jsonify


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("big.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    item_weight = float(request.form['item_weight'])
    item_fat_content = str(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = str(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_size = str(request.form['outlet_size'])
    outlet_location_type = str(request.form['outlet_location_type'])
    outlet_type = str(request.form['outlet_type'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    my_prediction = model.predict(X)
    r=my_prediction[0]

    return render_template("result.html", **locals())


if __name__ == "__main__":
    app.run(debug=True,port=7895)
