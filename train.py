import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import joblib

## Import dataset
leads_dataset = pd.read_csv('data/Leads_cleaned.csv')
leads_dataset.columns = map(str.lower, leads_dataset.columns)


# Create data pre-processing steps before plugging into model
leads_categorical_columns = ['lead origin', 'lead source', 'last activity', 'specialization', 'what is your current occupation', 'what matters most to you in choosing a course', 'city', 'last notable activity']

leads_numeric_columns = ['totalvisits', 'total time spent on website', 'page views per visit']

leads_response_columns = ['converted']


#split data for training, remove extras

leads_x = leads_dataset.drop(leads_response_columns, axis=1)
leads_y = leads_dataset[leads_response_columns]

leads_x_train, leads_x_test, leads_y_train, leads_y_test = train_test_split(leads_x, leads_y, train_size=0.7, test_size=0.3, random_state=5050)


scaler = StandardScaler()
scaler = scaler.fit(leads_x_train[leads_numeric_columns])

def pre_process_leads_data(df, numeric_columns, categorical_columns, fitted_scaler, train_df_columns = None):
    ## create new df with selected columns
    df.columns = map(str.lower, df.columns)
    _df = df[list(set(numeric_columns + categorical_columns))].copy()
    
    ## scale the numeric columns with the pre-built scaler
    _df[numeric_columns] = fitted_scaler.transform(_df[numeric_columns])
         
    # First, make categorical text lowercase
    _df[categorical_columns] = _df[categorical_columns].apply(lambda x: x.str.lower())
    # Next, create one-hot-encoded variables, add to dataframe, drop old columns
    _df_dummies = pd.get_dummies(_df[categorical_columns], drop_first=True)
    _df = pd.concat([_df, _df_dummies], axis=1)
    _df.drop(categorical_columns, axis=1, inplace = True)

    if train_df_columns:
        _df = _df.reindex(columns=train_df_columns, fill_value=0)

    return _df

leads_x_train_clean = pre_process_leads_data(df = leads_x_train,
                                            numeric_columns = leads_numeric_columns,
                                            categorical_columns = leads_categorical_columns,
                                            fitted_scaler = scaler)

leads_x_test_clean = pre_process_leads_data(df = leads_x_test,
                                           numeric_columns = leads_numeric_columns,
                                           categorical_columns = leads_categorical_columns,
                                           fitted_scaler = scaler,
                                           train_df_columns = leads_x_train_clean.columns.tolist())

## Train the random forest model
num_estimators = 100
min_samples = 4

rf = RandomForestClassifier(n_estimators=num_estimators,
                            min_samples_split=min_samples)
rf.fit(leads_x_train_clean, leads_y_train.values.ravel())

leads_y_test_predicted = rf.predict(leads_x_test_clean)

accuracy = metrics.accuracy_score(leads_y_test, leads_y_test_predicted)
auc_score = metrics.roc_auc_score(leads_y_test, leads_y_test_predicted)

print(accuracy)
print(auc_score)
# print(leads_y_test_predicted)
# joblib.dump(rf, "lead_scoring.joblib")
# joblib.dump(scaler, "scaler.joblib")
# joblib.dump(leads_x_train_clean.columns.tolist(), "train_df_columns.joblib")

# loaded_rf = joblib.load("lead_scoring.joblib")
# print(loaded_rf)