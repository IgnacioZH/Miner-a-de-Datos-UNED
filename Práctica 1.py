import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import paired_ttest_5x2cv

Iris=pd.read_csv('iris.csv')  


Iris["variety"].value_counts()

Iris = Iris.sample(frac=1).reset_index(drop=True)


X = Iris.iloc[:,:-1]
y = Iris.iloc[:,-1]
 
seeds = [13, 51, 137, 24659, 347]

    # se ha creado este bucle para poder obtener la contengency table y aplicar el test de McNemar
A1t=[]
A2t=[] 
A3t=[]
A4t=[]
for seed in seeds:


  k = 2
  kf = KFold(n_splits=k, random_state=seed,shuffle=True)
  model = GaussianNB()
  model2 = KNeighborsClassifier(n_neighbors=7)

 
  for train_index , test_index in kf.split(X):
     
     
     X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
     y_train , y_test = y[train_index] , y[test_index]
     
     model.fit(X_train,y_train)
     pred_values = model.predict(X_test)
     
          
     model2.fit(X_train,y_train)
     pred_values2 = model2.predict(X_test)
     
            
                      # para el MNemar test
     CtDf = pd.DataFrame({'y_true':y_test, 'y_model1':pred_values,'y_model2':pred_values2})
     A1=CtDf[(CtDf["y_true"] == CtDf["y_model1"]) &   (CtDf["y_true"] == CtDf["y_model2"])]
     A2=CtDf[(CtDf["y_true"] == CtDf["y_model1"]) &   (CtDf["y_true"] != CtDf["y_model2"])]
     A3=CtDf[(CtDf["y_true"] != CtDf["y_model1"]) &   (CtDf["y_true"] == CtDf["y_model2"])]
     A4=CtDf[(CtDf["y_true"] != CtDf["y_model1"]) &   (CtDf["y_true"] != CtDf["y_model2"])]
     
     A1=len(A1); A2=len(A2); A3=len(A3); A4=len(A4)
     

     A1t.append(A1)
     A2t.append(A2)
     A3t.append(A3)
     A4t.append(A4)
     
     
     
     
# Primer test : t test      
     
t, p = paired_ttest_5x2cv(estimator1=model, estimator2=model2, X=X, y=y, random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)     
     
     

# Segundo test, el de McNemar.
A1=np.mean(A1t)
A2=np.mean(A2t)
A3=np.mean(A3t)
A4=np.mean(A4t)

table = [[A1, A2], [A3, A4]]
result = mcnemar(table, exact=True)
result.pvalue
 
 

