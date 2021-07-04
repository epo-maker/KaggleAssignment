import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib as joblib
import lime.lime_tabular as lime_tabular


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


test_data= pd.read_csv('../data/raw/test.csv')
train_data = pd.read_csv('../data/raw/train.csv')
df_analysis= train_data.drop(columns=["Id_old","Id"])
target = df_analysis.pop('TARGET_5Yrs')
train_test_split = joblib.load('../models/train_test_split.joblib')
X_train, X_valid, y_train, y_valid = train_test_split (df_analysis, target, test_size=0.2, random_state=8)
drop_columns = ColumnTransformer(remainder = 'passthrough',
                                 transformers=[('drop_columns', 'drop', ['FGM','FGA','3P Made','3PA','FTM','FTA','REB'])])
model_fit = Pipeline(steps=[('drop_columns', drop_columns),  ('scale_variables',StandardScaler()), 
                            ('best_logit', LogisticRegression(penalty = 'l2',
                                                              C = 0.00480719434510272,
                                                              solver = 'lbfgs',
                                                              class_weight = 'balanced'))])
fitted_log = model_fit.fit(X_train,y_train)

explainer = lime_tabular.LimeTabularExplainer(np.array(X_train), 
                                 mode='classification', 
                                  feature_names=X_train.columns,
                                 class_names=["Less than 5Yrs", "More than 5 Yrs"])

class NBACareerPredict:
    def __init__ (self, test_data):
        self.test_data_transform = test_data.drop(columns=["Id_old","Id"])
        self.prediction = pd.DataFrame (test_data["Id"])
        self.prediction["Prediction"] = model_fit.predict(self.test_data_transform)
        self.chart_data = test_data.drop(columns=["Id_old","Id","FGM","FGA","3P Made","3PA","FTM","FTA","REB"])
        self.chart_data['TARGET_5YRS'] = model_fit.predict(self.test_data_transform)
        self.chart_data['Prediction'] = np.nan
        self.chart_data['Prediction'][self.chart_data['TARGET_5YRS'] == 0] = 'Less than 5 Yrs'
        self.chart_data['Prediction'][self.chart_data['TARGET_5YRS'] == 1] = 'More than 5 Yrs'
        
        
    def list_predictions (self):
        return self.prediction
    
    def visualise_features (self):
        for i, col in enumerate(self.chart_data.columns):
            plt.figure(i)
            chart = sns.histplot(data=self.chart_data, x=self.chart_data[col], hue = 'Prediction')
        return chart
    
    def explain_prediction (self, i):
        exp = explainer.explain_instance(
            self.test_data_transform[test_data["Id"]==i].iloc[0],
            model_fit.predict_proba,top_labels=1,num_features=10)
        return exp.show_in_notebook(show_table=True)

RookiePredict = NBACareerPredict(test_data)
