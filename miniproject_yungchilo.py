# -*- coding: utf-8 -*-
"""
Unison home ownership investors mini project
Yungchi Lo
"""
#Download data from database
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
conn=pymysql.connect(
  host="unisonim-recruitments-cluster.cluster-ro-cef17qxjlavg.us-west-2.rds.amazonaws.com",
  database="AgencyData",
  user="newuser",
  password="unisonim2.0",
  port=3306,
  charset='utf8'
 )
rawdata=pd.read_sql("select * from CleanFreddieSample",conn)

#Draw stack bar chart of years and status
data1=rawdata[["first_pmt_date","status"]]
data1["first_pmt_date"]=data1["first_pmt_date"].str.slice(stop=4)
barchar1=data1.groupby(["first_pmt_date","status"]).size()
barchar1.unstack().plot(kind='bar',stacked=True)
plt.show()

interest=rawdata[["first_pmt_date","orig_ir"]]
interest["first_pmt_date"]=interest["first_pmt_date"].str.slice(stop=4)
interest["orig_ir"]=interest["orig_ir"].astype(float)
interestchar=interest.groupby("first_pmt_date")['orig_ir'].mean()
interestchar.plot()

barchar1_percent=barchar1.groupby(level=0).apply(lambda x:100*x/float(x.sum()))
barchar1_percent.unstack().plot(kind='bar',stacked=True)
plt.show()

#Draw stack bar chart of age and status
data2=rawdata[["age","status"]]
barchar2=data2.groupby(["age","status"]).size()
chart=barchar2.unstack().plot(kind='bar',stacked=True)
chart.set_xticks([0,50,100,150,200,250])
chart.set_xticklabels(['0','50','100','150','200','250'])
plt.show()

barchar2_percent=barchar2.groupby(level=0).apply(lambda x:100*x/float(x.sum()))
chart2=barchar2_percent.unstack().plot(kind='bar',stacked=True)
chart2.set_xticks([0,50,100,150,200,250])
chart2.set_xticklabels(['0','50','100','150','200','250'])
plt.show()

#calcualte the probabilities
print(len(data2[(data2["age"]<=60)&(data2["status"]=="Default")])/len(data2)) 
print(len(data2[(data2["age"]<=60)&(data2["status"]=="Prepay")])/len(data2))
print(len(data2[(data2["age"]>=60)&(data2["status"]=="Alive")])/len(data2))
print(len(data2[(data2["age"]>=60)&(data2["status"]=="Alive")])/len(data2[data2["age"]>=60]))
#Cleaing the Data 
dataset=rawdata.loc[:]
dataset["first_pmt_date"]=dataset["first_pmt_date"].str.slice(stop=4)
dataset["defaultsign"]=dataset.status.apply(lambda x: 1 if x=="Default" else 0 ) #Bolean shows whether the loan default or not
#We supposed to do some data wrangling here because the raw data use "999","99" as NaN in the data set
#But luckily this time the raw data does't include those figures.
#So all we need to do here is to delete all the NaN rows, and find space like "", "  " 
dataset.loc[dataset['first_time_ho_flag']=="","first_time_ho_flag"]=np.NaN
dataset_clean=dataset.drop(['state','zip','msa_code','first_pmt_date','status','loan_seq_num'],axis=1)
#we need to find out tricky space left in our data
dataset_clean=dataset_clean.replace(r'\s+',np.nan,regex=True)
dataset_clean.loc[dataset_clean['orig_dti']=="","orig_dti"]=np.NaN
dataset_clean=dataset_clean.dropna(axis=0,how='any')
                           
dataset_clean["orig_cltv"]=dataset_clean.orig_cltv.astype(float)
dataset_clean["orig_dti"]=dataset_clean.orig_dti.astype(float)
dataset_clean["orig_ir"]=dataset_clean.orig_ir.astype(float)
dataset_clean["orig_upb"]=dataset_clean.orig_upb.astype(float)
dataset_clean["first_time_ho_flag"]=dataset_clean.first_time_ho_flag.astype('category')
dataset_clean["occupancy_status"]=dataset_clean.occupancy_status.astype('category')
dataset_clean["loan_purpose"]=dataset_clean.loan_purpose.astype('category')
dataset_clean["prop_type"]=dataset_clean.prop_type.astype('category')

#The ideal method here to predict the default probability is logistic regression
#But my pc's computing power is really .....(sigh) so I finish this part in AWS with R Language
#So the following part in python would be a simple non-linear regression :Deafault Sign ~ Credit Score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset_clean[["credit_score"]],dataset_clean[["defaultsign"]],random_state=0)
pf = PolynomialFeatures(degree=2)
x_fit_poly = pf.fit_transform(X_train)
x_test_poly=pf.fit_transform(X_test)

lrModel = LinearRegression(normalize=True)
lrModel.fit(x_fit_poly,y_train)
print('Coefficients:',lrModel.coef_)
print('intercept:',lrModel.intercept_) 

#Test
y_pred = lrModel.predict(x_test_poly)
print(y_pred)
print(y_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

#R Code For Lasso And Random Forest are in the R markdown file.