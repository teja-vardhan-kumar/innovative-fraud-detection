from datetime import date, datetime, time
import requests
import json
import numpy as np
import pandas as pd


data =  [{
    "accountID": "A985156985579195",
    "browserLanguage": "en-AU",
    "cardType": "VISA",
    "cvvVerifyResult": "M",
    "digitalItemCount": 1,
    "ipCountryCode": "au",
    "ipPostcode": "3000",
    "ipState": "victoria",
    "isProxyIP": False,
    "localHour": 19,
    "paymentBillingCountryCode": "AU",
    "paymentBillingPostalCode": "3122",
    "paymentBillingState": "Victoria",
    "paymentInstrumentType": "CREDITCARD",
    "physicalItemCount": 0,
    "transactionAmount": 99.0,
    "transactionAmountUSD": 103.48965,
    "transactionCurrencyCode": "AUD",
    "transactionDate": 20130409,
    "transactionID": "5EAC1EBD-1428-4593-898E-F4B56BC3FA06",
    "transactionIPaddress": 121.219,
    "transactionTime": 95000
    }]

#data = df.iloc[0,:].to_json(orient='split')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        
        return super(NpEncoder, self).default(obj)


body = str.encode(json.dumps(data, cls=NpEncoder))

url = 'http://localhost:5000/predict'


headers = {'Content-Type':'application/json'}

try:
    req = requests.post(url, body, headers=headers)
    result = req.json()
    print(result)
except:
    print("The request failed with status code: " + str(req.status_code))