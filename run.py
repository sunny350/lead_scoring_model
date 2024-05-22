import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

rf = joblib.load("lead_scoring.joblib")
scaler = joblib.load("scaler.joblib")
train_df_columns = joblib.load("train_df_columns.joblib")


def pre_process_new_data(df, numeric_columns, categorical_columns, fitted_scaler, train_df_columns):
    df.columns = map(str.lower, df.columns)
    _df = df[list(set(numeric_columns + categorical_columns))].copy()

    _df[numeric_columns] = fitted_scaler.transform(_df[numeric_columns])

    _df[categorical_columns] = _df[categorical_columns].apply(lambda x: x.str.lower())
    _df_dummies = pd.get_dummies(_df[categorical_columns], drop_first=True)
    _df = pd.concat([_df, _df_dummies], axis=1)
    _df.drop(categorical_columns, axis=1, inplace=True)

    _df = _df.reindex(columns=train_df_columns, fill_value=0)

    return _df

def predict_new_lead(new_data):
    leads_categorical_columns = ['lead origin', 'lead source', 'last activity', 'specialization', 'what is your current occupation', 'what matters most to you in choosing a course', 'city', 'last notable activity']
    leads_numeric_columns = ['totalvisits', 'total time spent on website', 'page views per visit']

    new_data_preprocessed = pre_process_new_data(new_data,
                                                 leads_numeric_columns,
                                                 leads_categorical_columns,
                                                 scaler,
                                                 train_df_columns)
    predictions = rf.predict(new_data_preprocessed)
    probabilities = rf.predict_proba(new_data_preprocessed)[:, 1]
    return predictions, probabilities


# new_data_dict = {
#     'lead origin': ['Landing Page Submission'],
#     'lead source': ['Google'],
#     'last activity': ['Email Opened'],
#     'specialization': ['Management'],
#     'what is your current occupation': ['Unemployed'],
#     'what matters most to you in choosing a course': ['Better Career Prospects'],
#     'city': ['Mumbai'],
#     'last notable activity': ['Email Opened'],
#     'totalvisits': [5],
#     'total time spent on website': [246],
#     'page views per visit': [2.5],
# }

new_data_dict = {
    'lead origin': ['Landing Page Submission'],
    'lead source': ['Google'],
    'last activity': ['Email Opened'],
    'specialization': ['Management'],
    'what is your current occupation': ['Working Professional'],
    'what matters most to you in choosing a course': ['Better Career Prospects'],
    'city': ['Mumbai'],
    'last notable activity': ['SMS Sent'],
    'totalvisits': [15],
    'total time spent on website': [800],
    'page views per visit': [10],
}

new_data_df = pd.DataFrame(new_data_dict)

predictions, probabilities = predict_new_lead(new_data_df)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Lead {i+1}: Prediction - {'Will Buy' if pred == 1 else 'Will Not Buy'}, Probability - {prob:.2f}")
    
    
