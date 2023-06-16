import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from customestimators import NumericCleaner, CategoricalCleaner


# sklearn.externals.joblib was deprecated in 0.21
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.21.0"):
    from sklearn.externals import joblib
else:
    import joblib
    

account_df = pd.read_csv('./data/Account_Info.csv')
fraud_df = pd.read_csv('./data/Fraud_Transactions.csv')
untagged_df = pd.read_csv( './data/Untagged_Transactions.csv')

account_df_clean = account_df[["accountID", "transactionDate", "transactionTime", 
                               "accountPostalCode", "accountState", "accountCountry", 
                               "accountAge", "isUserRegistered", "paymentInstrumentAgeInAccount", 
                               "numPaymentRejects1dPerUser"]]
account_df_clean = account_df_clean.copy()
account_df_clean['paymentInstrumentAgeInAccount'] = pd.to_numeric(account_df_clean['paymentInstrumentAgeInAccount'], errors='coerce')
account_df_clean['paymentInstrumentAgeInAccount'] = account_df_clean[['paymentInstrumentAgeInAccount']].fillna(0)['paymentInstrumentAgeInAccount']
account_df_clean["numPaymentRejects1dPerUser"] = account_df_clean[["numPaymentRejects1dPerUser"]].astype(float)["numPaymentRejects1dPerUser"]


untagged_df_clean = untagged_df.dropna(axis=1, how="all").copy()
untagged_df_clean["localHour"] = untagged_df_clean["localHour"].fillna(-99)
untagged_df_clean.loc[untagged_df_clean.loc[:,"localHour"] == -1, "localHour"] = -99

untagged_df_clean = untagged_df_clean.fillna(value={"ipState": "NA", "ipPostcode": "NA", "ipCountryCode": "NA", 
                               "isProxyIP":False, "cardType": "U", 
                               "paymentBillingPostalCode" : "NA", "paymentBillingState":"NA",
                               "paymentBillingCountryCode" : "NA", "cvvVerifyResult": "N"
                              })
del untagged_df_clean["transactionScenario"]
del untagged_df_clean["transactionType"]

fraud_df_clean = fraud_df.copy()
fraud_df_clean["localHour"] = fraud_df_clean["localHour"].fillna(-99)
del fraud_df_clean['transactionDeviceId']

all_labels = untagged_df_clean["transactionID"].isin(fraud_df_clean["transactionID"])
all_transactions = untagged_df_clean



numeric_features=["transactionAmountUSD", "transactionDate", "transactionTime", "localHour", 
                  "transactionIPaddress", "digitalItemCount", "physicalItemCount"]

categorical_features=["transactionCurrencyCode", "browserLanguage", "paymentInstrumentType", "cardType", "cvvVerifyResult"]                           

numeric_transformer = Pipeline(steps=[
    ('cleaner', NumericCleaner()),
    ('scaler', StandardScaler())
])
                               
categorical_transformer = Pipeline(steps=[
    ('cleaner', CategoricalCleaner()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

preprocessed_result = preprocessor.fit_transform(all_transactions)


only_fraud_samples = all_transactions.loc[all_labels == True]
only_fraud_samples["label"] = True
only_non_fraud_samples = all_transactions.loc[all_labels == False]
only_non_fraud_samples["label"] = False
random_non_fraud_samples = only_non_fraud_samples.sample(n=1151, replace=False, random_state=42)
balanced_transactions = pd.concat([random_non_fraud_samples, only_fraud_samples])


balanced_labels = balanced_transactions["label"]
del balanced_transactions["label"]

X_train, X_test, y_train, y_test = train_test_split(balanced_transactions, balanced_labels, 
                                                    test_size=0.2, random_state=42)



svm_clf = Pipeline((
    ("preprocess", preprocessor),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
))
svm_clf.fit(X_train, y_train)
y_test_preds = svm_clf.predict(X_test)
print(confusion_matrix(y_test, y_test_preds))
print(accuracy_score(y_test, y_test_preds))
print("Accuracy:", accuracy_score(y_test, y_test_preds))
print("Precision:", precision_score(y_test, y_test_preds))
print("Recall:", recall_score(y_test, y_test_preds))
print("F1:", f1_score(y_test, y_test_preds))
print("AUC:", roc_auc_score(y_test, y_test_preds))

p
joblib.dump(svm_clf, 'fraud_score.pkl')
print("model saved")