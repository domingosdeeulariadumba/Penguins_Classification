# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:50:42 2023

@author: domingosdeeularia
"""


# %%
""" IMPORTING LIBRARIES """


import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer as SI
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
from sklearn.metrics import roc_auc_score as ras, roc_curve as rcv, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler as SS, normalize, LabelBinarizer as LB

from itertools import cycle

import joblib as jbl

import warnings
warnings.filterwarnings('ignore')
# %%



# %%
"""" EXPLORATORY DATA ANALYSIS """


    '''
    Importing the dataset
    '''

df = pd.read_csv('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/penguins_dataset.csv')


    '''
    Displaying the information of the dataset (including the first and last 
    ten records)
    '''
df.info()
df.head()
df.tail()


    '''
    From the information above, it is noticed that there are some missing
    entries. It is fixed below, using Most Frequent Strategy.
    '''
df.iloc[:,:] = SI(strategy = 'most_frequent').fit_transform(df)


    '''
    Statistical summary
    '''
df.describe(include = 'all')


    '''
    It was found out that, after imputation, there are still more than two 
    entries for 'sex'. This one ('.') will be replaced with the most frequent 
    sex ('MALE').
    '''
df['sex'].value_counts()
df['sex'][df['sex']=='.'] = 'MALE'


    '''
    Next is presented the visual summary (for numeric entries only).
    '''
sb.pairplot(df, hue = 'species')
plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/pairplot.png')
plt.close()


    '''
    How are the non-numeric variables distributed?
    '''
for i in list(df[['species', 'island', 'sex']].columns):
    plt.figure()
    sb.displot(x = df[i])
    plt.xticks(rotation = 16)
    plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/{0}.displot.png'.format(i))
    plt.close()

     '''
     How are the independent variables related to the dependent one (species)?
     '''
for i in list(df.drop('species', axis = 1).columns):
    plt.figure()
    sb.stripplot(x = i, y = 'species', data=df)
    plt.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/{0}.stripplot.png".format(i))
    plt.close()


     '''
     What are on average the main characteristics of the penguin species in
     case?
     '''
df.groupby('species').mean()
#%%



# %%
"""" BUILDING THE CLASSIFICATION MODEL """



    '''
    Since there are categorical independent variables between. It will be first
    hot encoded.
    We next convert the different outcome entries to numeric values.
    '''
x,y = pd.get_dummies(df.drop('species', axis = 1),
                     drop_first = True),df['species'].copy()
   
for i in range(len(df)):
    if y[i] == 'Adelie':
        y[i] = 0
    elif y[i] == 'Gentoo':
        y[i] = 1
    else:
        y[i] = 2

y = y.astype(int)


    '''
    As the pair plot demonstrates, the attributes are not normally distributed.
    To address this issue, below we normalize these entries (and then also
    scale and eliminate irrelevant predictors features). 
    '''
def NSRFE(x):
    
    
    # Normalization
    X_norm = normalize(x)
        
    # Scaling   
    X_scld = SS().fit_transform(X_norm)
        
    # Recursive Feature Elimination, RFE                    
    rfe = RFE(LogR(), n_features_to_select = None)
    rfe_fit = rfe.fit(x,y)
             
    
    return x[x.columns[rfe.support_.tolist()]]

X = NSRFE(x)


    '''
    Displaying the final features.
    '''
for i in list(X.columns):
    print(i)


    '''
    Splitting data into train and test sets
    '''
X_train, X_test, y_train, y_test = TTS(X,y, test_size = 0.2, random_state = 33)


    '''
    Creating One vs Rest (OvR) classification model. Prior to this step it is 
    displayed the Confusion Matrix and the total outputs for each combination
    and then the model is trained.
    '''
model = LogR(multi_class = 'ovr').fit(X_train, y_train)
y_pred = model.predict(X_test)
print ('Accuracy of the model:', round(model.score(X_test, y_test),2))
print ('Classification Report:', cr(y_test, y_pred))
print ('Confusion Matrix:', cm(y_test, y_pred))

def DisplayTotal(CM):
    
    CM = pd.DataFrame(cm(y_test, y_pred))
    TruePositives = 0
    TrueNegatives = 0
    FalsePositives = 0
    FalseNegatives = 0
   
    Gen_CM = pd.DataFrame({'Class': [], 'True Positives': [], 'True Negatives': [],
                           'False Negatives': [], 'False Positives': []})
  
    for row,column in zip(range(len(CM)),range(len(CM))):    
       if row == column:
           TruePositives = CM[row][column]
           TrueNegatives = CM.drop(row, axis =0).drop(column, axis =1).sum().sum()
           FalsePositives = CM[row].sum()-CM[row][row]
           FalseNegatives = CM.loc[row].sum()-CM[row][column]
           
           Gen_CM = Gen_CM.append({'Class': row, 'True Positives':TruePositives, 
                     'True Negatives':TrueNegatives,
                     'False Negatives':FalseNegatives, 
                     'False Positives':FalsePositives},
                                  ignore_index = True).astype(int)
    
    return display(Gen_CM)

DisplayTotal(cm(y_test, y_pred))


    '''
    The model has presented a performance of about 83%, with 57 out of 69 
    correct classifications for Adelie and Chinstrap species of all the test 
    set and 69 for Gentoo specie. We'll then binarize the classes to get the 
    ROC curves and proceed with the performance analysis. As stated in the 
    sklearn documentation, "ROC curves are typically used in binary 
    classification, where the TPR and FPR can be defined unambiguously. In the
    case of multiclass classification, a notion of TPR or FPR is obtained only
    after binarizing the output." This can be done in two different ways:
    1) the One-vs-Rest (OvR) scheme, which compares each class against all the
    others (assumed as one); and 2) the One-vs-One (OvO) scheme, that compares
    every unique pairwise combination of classes.
    '''
l_bin = LB().fit(y_train)

y_bin_test = l_bin.transform(y_test)

y_pred_prb = model.predict_proba(X_test)

print('Encoding of each class:', l_bin.transform([0,1,2]))


    '''
    The strategy we'll adopte is the OvR, below we compute the weighted-average
    ROC-AUC considering this approach and the one with the overall performance.
    '''
fpr, tpr, roc_auc = dict(), dict(), dict()

fpr["weighted"], tpr["weighted"], _ = rcv(y_bin_test.ravel(),
                                          y_pred_prb.ravel())

roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])


    '''
    ROC-AUC plot
    '''
classes = len(np.unique(y))
target_names = list(np.unique(y))
colors = cycle(["b", "r", "k"])

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(
    fpr["weighted"], tpr["weighted"],
    label=f"weighted-average ROC curve (AUC = {roc_auc['weighted']:.2f})",
    color="m", linestyle="-.", linewidth=4)
for i, color in zip(range(classes), colors):
    RocCurveDisplay.from_predictions(
        y_bin_test[:, i], y_pred_prb[:, i],
        name=f"ROC curve for {target_names[i]}",
        color=color,ax=ax)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of ROC-AUC to OvR multiclass")
plt.legend()
plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/AUC-ROC_CURVEpenguins.png')
plt.close()


    '''
    We confirm that the penguin species were very well identified by a linear
    classifier with individual scores starting from 88% (for Chinstrap) whereas
    the overall performance is 97%. Below the model is saved using Joblib.
    '''
jbl.dump(model,"C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/Penguins_Classification/penguinsClass.sav")
# %%
