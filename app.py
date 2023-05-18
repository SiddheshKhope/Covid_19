# # To check if the loaded model is okay
# import pickle
# with open("model_covid.pkl", "rb") as f:
#     model = pickle.load(f)

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import pandas as pd
import numpy as np
import pickle

class DataType(BaseModel):
    sex: int
    patient_type: int
    intubed: int
    pneumonia: int
    age: int
    pregnancy: int
    diabetes: int
    copd: int
    asthma: int
    inmsupr: int
    hypertension: int
    other_disease: int
    cardiovascular: int
    obesity: int
    renal_chronic: int
    tobacco: int
    contact_other_covid: int
    icu: int



app = FastAPI()

"""
{
    "sex": 2,
    "patient_type": 1,
    "intubed": 97,
    "pneumonia": 2,
    "age": 27,
    "pregnancy": 97,
    "diabetes": 2,
    "copd": 2,
    "asthma": 2,
    "inmsupr": 2,
    "hypertension": 2,
    "other_disease": 2,
    "cardiovascular": 2,
    "obesity": 2,
    "renal_chronic": 2,
    "tobacco": 2,
    "contact_other_covid": 2,
    "icu": 97
}



"""


with open("model_covid.pkl", "rb") as f:
    model = pickle.load(f)
    
def scale_data(df):
    count_class_1, count_class_0 = df.Chance.value_counts()

    # Divide by class
    df_class_0 = df[df['Chance'] == 0]
    df_class_1 = df[df['Chance'] == 1]
    df_class_0_over = df_class_0.sample(count_class_1+100,replace=True)
    df_test_over = pd.concat([df_class_1,df_class_0_over],axis=0)

@app.post("/predict")
async def predict(item: DataType):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    # data = scale_data(df)
    print(df)
    finalmod = model.predict(df)
    if finalmod[0] == 0:
        return "Covid Negative"
    else:
        return "Covid Positive"
    # df = one_hot_encoder(df)
    # print(df)
    # print(df.shape)
    # ans1 = model.predict(data)
    # ans1 = list(ans1)
    # if ans1[0] == 0:
    #     return "Benign Prostatic Hyperplasia (BPH)"
    # else:
    #     return "Malignant Prostate Cancer (MPC)"

@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
