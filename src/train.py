#!/usr/bin/env python
# coding: utf-8

# <center> <h1> Assignment 6 (ML Operations) </h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center><h4> Predict the quality of red wine from its physico-chemical properties </h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# ---

# <a id='P0' name="P0"></a>
#   <ol>
#       <li> <a style="color:#303030" href='#SU'>Set up</a></li>
#       <li> <a style="color:#303030" href='#P1'>Loading Data and Train-Test Split</a></li>
#       <li> <a style="color:#303030" href='#P2'>Modelling</a></li>
#       <li> <a style="color:#303030" href='#P3'>Model Evluation and Explainability</a></li>
#   </ol>

# <a id='SU' name="SU"></a>
# ### [Set up](#P0)

# #### packages install

# In[1]:

# #### Packages imports

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import autosklearn.regression
import PipelineProfiler

import plotly.express as px
import plotly.graph_objects as go

from joblib import dump

import shap

import datetime

import logging

import matplotlib.pyplot as plt


# #### Google Drive connection

# In[2]:

# #### options and settings

# In[3]:


data_path = "/content/drive/MyDrive/SIT/6_Mohammad_Nadeem_Ghafoori_Assignment/Assignment_6/data/raw/"


# In[4]:


model_path = "/content/drive/MyDrive/SIT/6_Mohammad_Nadeem_Ghafoori_Assignment/Assignment_6/models/"


# In[5]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[6]:


log_config = {
    "version":1,
    "root":{
        "handlers" : ["console"],
        "level": "DEBUG"
    },
    "handlers":{
        "console":{
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters":{
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt":"%d-%m-%Y %I:%M:%S"
        }
    },
}


# In[7]:


logging.config.dictConfig(log_config)


# <a id='P1' name="P1"></a>
# ### [Loading Data and Train-Test Split](#P0)
# 

# In[8]:


df = pd.read_csv(f'{data_path}winequality-red.csv')


# In[9]:


test_size = 0.2
random_state = 0


# In[10]:


train, test = train_test_split(df, test_size=test_size, random_state=random_state)


# In[11]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[12]:


train.to_csv(f'{data_path}winequality-red.csv', index=False)


# In[13]:


train = train.copy()


# In[14]:


test.to_csv(f'{data_path}winequality-red.csv', index=False)


# In[15]:


test = test.copy()


# <a id='P2' name="P2"></a>
# ### [Modelling](#P0)

# In[16]:


X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# In[17]:


total_time = 60
per_run_time_limit = 30


# In[18]:


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
    logging_config=log_config
)
automl.fit(X_train, y_train)


# In[19]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[20]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[21]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[22]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# <a id='P3' name="P3"></a>
# ### [Model Evaluation and Explainability](#P0)

# In[ ]:


X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# #### Model Evaluation

# In[ ]:


y_pred = automl.predict(X_test)


# In[ ]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# In[ ]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
              'pH', 'sulphates', 'alcohol', 'True Target', 'Predicted Target']


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")

