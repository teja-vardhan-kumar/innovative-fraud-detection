import pandas as pd

# sklearn.externals.joblib was deprecated in 0.21
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.21.0"):
    from sklearn.externals import joblib
else:
    import joblib

desired_cols = ['accountID',
 'browserLanguage',
 'cardType',
 'cvvVerifyResult',
 'digitalItemCount',
 'ipCountryCode',
 'ipPostcode',
 'ipState',
 'isProxyIP',
 'localHour',
 'paymentBillingCountryCode',
 'paymentBillingPostalCode',
 'paymentBillingState',
 'paymentInstrumentType',
 'physicalItemCount',
 'transactionAmount',
 'transactionAmountUSD',
 'transactionCurrencyCode',
 'transactionDate',
 'transactionID',
 'transactionIPaddress',
 'transactionTime']

scoring_pipeline = joblib.load('fraud_score.pkl')

untagged_df_fresh = pd.read_csv('./data/Untagged_Transactions.csv')[desired_cols]

test_pipeline_preds = scoring_pipeline.predict(untagged_df_fresh)

print(test_pipeline_preds)