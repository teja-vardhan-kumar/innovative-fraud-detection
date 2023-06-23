import logging
import warnings
import pandas as pd
from flask import Flask, request, jsonify
from flask.logging import create_logger
warnings.filterwarnings("ignore")


# sklearn.externals.joblib was deprecated in 0.21
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.21.0"):
    from sklearn.externals import joblib
else:
    import joblib

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

# loading pickle file
clf = joblib.load("fraud_score.pkl")


@app.route("/")
def home():
    html = "<h3>Fraud Detection Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    
    json_payload = request.json
    print(json_payload)
    LOG.info("JSON payload: %s json_payload")
    payload = pd.DataFrame(json_payload)
    LOG.info("inference payload DataFrame: %s payload")
    
    prediction = clf.predict(payload).tolist()
    
    return jsonify({'Fraud': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 