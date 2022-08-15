import pickle
from flask import Flask,request,jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def data_preprocessing(data):
    raw_df = pd.read_csv('train.csv')
    raw_df.set_index('PassengerId',inplace=True)
    raw_df.drop(['Name','Ticket', 'Cabin'],inplace=True,axis=1)
    raw_df = raw_df.append(data,ignore_index=True)
    sex_dummy = pd.get_dummies(raw_df['Sex'])
    embarked_dummy = pd.get_dummies(raw_df["Embarked"])
    merged = pd.concat([raw_df, sex_dummy, embarked_dummy], axis='columns')
    merged.loc[merged.Fare <= 7.91, 'Fare'] = 0
    merged.loc[(merged.Fare > 7.91) & (merged.Fare <= 14.45), 'Fare'] = 1
    merged.loc[(merged.Fare > 14.54) & (merged.Fare <= 31), 'Fare'] = 2
    merged.loc[(merged.Fare > 31), 'Fare'] = 3
    merged.loc[merged.Age <= 16, 'Age'] = 0
    merged.loc[(merged.Age > 16) & (merged.Age <= 32), 'Age'] = 1
    merged.loc[(merged.Age > 32) & (merged.Age <= 48), 'Age'] = 2
    merged.loc[(merged.Age > 48) & (merged.Age <= 64), 'Age'] = 3
    merged.loc[(merged.Age > 64), 'Age'] = 4
    merged.drop(['Sex', 'Embarked','Survived'], inplace=True, axis=1)
    x = merged.iloc[-1]
    model = pickle.load(open('TitanicDisaster.ipynb', 'rb'))
    pred = model.predict([x])[0]
    return pred
@app.route('/',methods=["POST"])
def index():
    data = request.json
    pred = data_preprocessing(data)

    return jsonify(int(pred))
if __name__=="__main__":
    app.run(debug=True)