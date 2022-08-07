import pickle
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from flask import Flask,request,jsonify

app = Flask(__name__)


def data_processing(data):
    raw_df = pd.read_csv('weatherAUS.csv')
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = raw_df.select_dtypes('object').columns.tolist()[1:-1]
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(raw_df[numeric_cols])
    scaler = MinMaxScaler()
    scaler.fit(raw_df[numeric_cols])
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(raw_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_input = data
    new_input_df = pd.DataFrame([new_input])
    new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
    new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
    new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
    x_new_input = new_input_df[numeric_cols + encoded_cols]
    x_new_input = x_new_input.drop(['RainToday_nan'], axis=1)
    model = pickle.load(open('RainTomorrowUsingLogisticRegression (8).ipynb', 'rb'))
    pred = model.predict(x_new_input)[0]
    prob = model.predict_proba(x_new_input)[0][list(model.classes_).index(pred)]
    return pred, prob


@app.route("/",methods=["POST"])
def hello_world():
    data = request.json
    res = {}
    res["output"] = data_processing(data)
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)
