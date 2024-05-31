from sklearn.svm import SVC  #Support Vector Machines(svm) & Classification(svc)
from sklearn.feature_selection import RFE #Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn import preprocessing, model_selection, svm, neighbors
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"] 

df = pd.read_csv('data.csv')
df.columns = ['having_IP_Address','URL_Length','Shortening_Service','having_At_Symbol','double_slash_redirecting',
'Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registration_length','Favicon','port','HTTPS_token',
'Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover',
'RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic ','Page_Rank','Google_Index',
'Links_pointing_to_page','Statistical_report','Result']

names = df.head()
X = df[df.columns[:-1]].values
y = df[['Result']].values

svc = SVC(kernel="linear", C=1)  #To convert n-dimentional data to linear data by obtaining the dot product of the vectors
#Kernel describes the hyper-plane. 'rbf' can lead to overfitting (radial basis function)
# Penalty parameter C of the error term.
#It also controls the trade off between smooth decision boundary and classifying the training points correctly.
rfe = RFE(estimator=svc, n_features_to_select=3, step=1)
# estimator assigns weights to features
#step corresponds to the number of features to remove at each iteration
#Fit the RFE model and then the underlying estimator on the selected features.
rfe.fit(X, y.ravel()) 

XX=[]
YY=[]
mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
for i in mm:
    XX.append(i[0])
    YY.append(i[1])

plt.bar(YY[:3],XX[:3],align='center', alpha=0.5,color=colors)
plt.xlabel('Feature Selection')
plt.ylabel('RANK')
plt.title("Feature Selection BY SVM");
plt.show()

cols=[]
cols.append(mm[0][1])
cols.append(mm[1][1])
cols.append(mm[2][1])

sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

X_new = rfe.transform(X)  #Reduce X to the selected features.
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)
svc.fit(X_train, y_train.ravel())
ysvc = svc.predict(X_test)

mse=[]
mae=[]

rsq=[]
rmse=[]
acy=[]

print("MSE VALUE FOR SVM IS %f "  % mean_squared_error(y_test,ysvc))
print("MAE VALUE FOR SVM IS %f "  % mean_absolute_error(y_test,ysvc))
print("R-SQUARED VALUE FOR SVM IS %f "  % r2_score(y_test,ysvc))
rms = np.sqrt(mean_squared_error(y_test,ysvc))
print("RMSE VALUE FOR SVM IS %f "  % rms)
ac=accuracy_score(y_test,ysvc) * 100
print ("ACCURACY VALUE SVM IS %f" % ac)

mse.append(mean_squared_error(y_test,ysvc))
mae.append(mean_absolute_error(y_test,ysvc))
rsq.append(r2_score(y_test,ysvc))
rmse.append(rms)
acy.append(ac)

lr = LogisticRegression(solver = 'lbfgs')
# create the RFE model and select 3 attributes
rfe = RFE(lr, 3)
rfe = rfe.fit(X,y.ravel())

mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

a1=mm[0]
a2=mm[1]
a3=mm[2]


XX=[]
YY=[]
XX.append(a1[1])
YY.append(1)
XX.append(a2[1])
YY.append(1)
XX.append(a3[1])
YY.append(1)
    
#Barplot for the dependent variable
fig = plt.figure(0)
plt.bar(XX,YY,align='center', alpha=0.5,color=colors)
plt.xlabel('Feature Selection')
plt.ylabel('RANK')
plt.title("Feature Selection BY LogisticRegression");

plt.show()

cols=[]

cols.append(a1[1])
cols.append(a2[1])
cols.append(a3[1])

sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

X_new = rfe.transform(X) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y, test_size = 0.25)
lr.fit(X_train, y_train.ravel())
ylr = lr.predict(X_test)

print("MSE VALUE FOR LogisticRegression IS %f "  % mean_squared_error(y_test,ylr))
print("MAE VALUE FOR LogisticRegression IS %f "  % mean_absolute_error(y_test,ylr))
print("R-SQUARED VALUE FOR LogisticRegression IS %f "  % r2_score(y_test,ylr))
rms = np.sqrt(mean_squared_error(y_test,ylr))
print("RMSE VALUE FOR LogisticRegression IS %f "  % rms)
ac=accuracy_score(y_test,ylr) * 100
print ("ACCURACY VALUE LogisticRegression IS %f" % ac)
mse.append(mean_squared_error(y_test,ylr))
mae.append(mean_absolute_error(y_test,ylr))
rsq.append(r2_score(y_test,ylr))
rmse.append(rms)
acy.append(ac)

dtree = tree.DecisionTreeClassifier()
rfe = RFE(estimator=dtree, n_features_to_select=3)
rfe.fit(X, y.ravel())

mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
a1=mm[0]
a2=mm[1]
a3=mm[2]


XX=[]
YY=[]

XX.append(a1[1])
YY.append(1)
XX.append(a2[1])
YY.append(1)
XX.append(a3[1])
YY.append(1)

#Barplot for the dependent variable
fig = plt.figure(0)
plt.bar(XX,YY,align='center', alpha=0.5,color=colors)
plt.xlabel('Feature Selection')
plt.ylabel('RANK')
plt.title("Feature Selection BY DecisionTree");
plt.show()
    
cols=[]
cols.append(a1[1])
cols.append(a2[1])
cols.append(a3[1])

sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()


X_new = rfe.transform(X) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y, test_size = 0.25)
dtree.fit(X_train, y_train)
ydtree = dtree.predict(X_test)
print("MSE VALUE FOR DecisionTree IS %f "  % mean_squared_error(y_test,ydtree))
print("MAE VALUE FOR DecisionTree IS %f "  % mean_absolute_error(y_test,ydtree))
print("R-SQUARED VALUE FOR DecisionTree IS %f "  % r2_score(y_test,ydtree))
rms = np.sqrt(mean_squared_error(y_test,ydtree))
print("RMSE VALUE FOR DecisionTree IS %f "  % rms)
ac=accuracy_score(y_test,ydtree) * 100
print ("ACCURACY VALUE DecisionTree IS %f" % ac)
mse.append(mean_squared_error(y_test,ydtree))
mae.append(mean_absolute_error(y_test,ydtree))

rsq.append(r2_score(y_test,ydtree))
rmse.append(rms)
acy.append(ac)

rf = RandomForestClassifier(n_estimators = 10)
rfe = RFE(estimator=rf, n_features_to_select=3)
rfe.fit(X, y.ravel())

mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
a1=mm[0]
a2=mm[1]
a3=mm[2]

XX=[]
YY=[]
XX.append(a1[1])
YY.append(1)
XX.append(a2[1])
YY.append(1)
XX.append(a3[1])
YY.append(1)

#Barplot for the dependent variable
fig = plt.figure(0)
plt.bar(XX,YY,align='center', alpha=0.5,color=colors)
plt.xlabel('Feature Selection')
plt.ylabel('RANK')
plt.title("Feature Selection BY RandomForest");
plt.show()

cols=[]
cols.append(a1[1])
cols.append(a2[1])
cols.append(a3[1])

sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()
 
X_new = rfe.transform(X) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y, test_size = 0.25)

rf.fit(X_train, y_train.ravel())
yrf = rf.predict(X_test)

print("MSE VALUE FOR Random Forest IS %f "  % mean_squared_error(y_test,yrf))
print("MAE VALUE FOR Random Forest IS %f "  % mean_absolute_error(y_test,yrf))
print("R-SQUARED VALUE FOR Random Forest IS %f "  % r2_score(y_test,yrf))
rms = np.sqrt(mean_squared_error(y_test,yrf))
print("RMSE VALUE FOR Random Forest IS %f "  % rms)
ac=accuracy_score(y_test,yrf) * 100
print ("ACCURACY VALUE Random Forest IS %f" % ac)
mse.append(mean_squared_error(y_test,yrf))
mae.append(mean_absolute_error(y_test,yrf))
rsq.append(r2_score(y_test,yrf))
rmse.append(rms)
acy.append(ac)

knn = KNeighborsClassifier(n_neighbors=4)
sfs1 = SFS(knn,k_features=3,forward=True,floating=False,verbose=2,scoring='accuracy',cv=0)
sfs1 = sfs1.fit(X, y.ravel())
print("-----------------")
print(sfs1.k_feature_idx_)
XX=[]
YY=[]
for kk in sfs1.k_feature_idx_:
print(kk)
print(df.columns[kk])
XX.append(df.columns[kk])
YY.append(1)
print(sfs1.k_score_)

#Barplot for the dependent variable
fig = plt.figure(0)
plt.bar(XX,YY,align='center', alpha=0.5,color=colors)
plt.xlabel('Feature Selection')
plt.ylabel('RANK')
plt.title("Feature Selection BY KNN");
plt.show()
    
cols=[]
cols.append(a1[1])
cols.append(a2[1])
cols.append(a3[1])

sns.set(style='whitegrid', context='notebook')

sns.pairplot(df[cols], height=1.5);
plt.show()

cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()


X_new = rfe.transform(X) 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y, test_size = 0.25)
knn.fit(X_train, y_train.ravel())
yknn = knn.predict(X_test)
print(yknn)
accuracy = 100.0 * accuracy_score(y_test, yknn)
print ("The accuracy is: " + str(accuracy))

print("MSE VALUE FOR KNN IS %f "  % mean_squared_error(y_test,yknn))
print("MAE VALUE FOR KNN IS %f "  % mean_absolute_error(y_test,yknn))
print("R-SQUARED VALUE FOR KNN IS %f "  % r2_score(y_test,yknn))
rms = np.sqrt(mean_squared_error(y_test,yknn))
print("RMSE VALUE FOR KNN IS %f "  % rms)
ac=accuracy_score(y_test,yknn) * 100
print ("ACCURACY VALUE KNN IS %f" % ac)
mse.append(mean_squared_error(y_test,yknn))
mae.append(mean_absolute_error(y_test,yknn))
rsq.append(r2_score(y_test,yknn))
rmse.append(rms)
acy.append(ac)

al = ['SVM','KNN','RF','LR','DT']
l = np.arange(len(al))

plt.bar(l,mse,align='center', alpha=0.5,color=colors)
plt.xticks(l, al)
plt.xlabel('Algorithm')
plt.ylabel('MSE')
plt.title("MSE Value");
plt.show()
    
plt.bar(l,mae,align='center', alpha=0.5,color=colors)
plt.xticks(l, al)
plt.xlabel('Algorithm')
plt.ylabel('MAE')
plt.title('MAE Value')
plt.show()

plt.bar(l,rsq,align='center', alpha=0.5,color=colors)
plt.xticks(l, al)

plt.xlabel('Algorithm')
plt.ylabel('R-SQUARED')
plt.title('R-SQUARED Value')
plt.show()
      
plt.bar(l, rmse, align='center', alpha=0.5,color=colors)
plt.xticks(l, al)
plt.xlabel('Algorithm')
plt.ylabel('RMSE')
plt.title('RMSE Value')
plt.show()
    
plt.bar(l, acy,align='center', alpha=0.5,color=colors)
plt.xticks(l, al)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Value')
plt.show()
