#!/usr/bin/env python
# coding: utf-8

# Using Test Train Split

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("Dataset.csv")
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[2]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

model = RandomForestClassifier(max_depth=2, random_state=0)

skf = StratifiedKFold(n_splits=10)
scores = []
precisions = []
specificities = []
sensitivities = []
recalls = []
balanced_accuracies = []
f1_scores = []
confusion_matrices = []

for k, (train_index, test_index) in enumerate(skf.split(X_train, Y_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    Y_train_fold, Y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    model.fit(X_train_fold, Y_train_fold)
    y_pred = model.predict(X_test_fold)
    
    score = accuracy_score(Y_test_fold, y_pred)
    precision = precision_score(Y_test_fold, y_pred)
    recall = recall_score(Y_test_fold, y_pred)
    balanced_accuracy = balanced_accuracy_score(Y_test_fold, y_pred)
    f1 = f1_score(Y_test_fold, y_pred)
    confusion = confusion_matrix(Y_test_fold, y_pred)
    
    scores.append(score)
    precisions.append(precision)
    recalls.append(recall)
    balanced_accuracies.append(balanced_accuracy)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)
    
    tn, fp, fn, tp = confusion.ravel()
    specificity = tn / (tn + fp)
    specificities.append(specificity)
    sensitivity = tp / (tp + fn)
    sensitivities.append(sensitivity)

    print('Fold: %2d' % (k+1))
    print('Accuracy: %.3f, Precision: %.3f, Specificity: %.3f, Sensitivity: %.3f, Recall: %.3f' % (
        score, precision, specificity, sensitivity, recall))
    print('Balanced Accuracy: %.3f, F1 Score: %.3f' % (balanced_accuracy, f1))
    print('Confusion Matrix:\n', confusion)
    print('\n')

mean_score = np.mean(scores)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_balanced_accuracy = np.mean(balanced_accuracies)
mean_f1 = np.mean(f1_scores)
mean_specificity = np.mean(specificities)
mean_sensitivity = np.mean(sensitivities)
mean_confusion = np.mean(confusion_matrices, axis=0)

print('Mean Accuracy: %.3f, Precision: %.3f, Specificity: %.3f, Sensitivity: %.3f, Recall: %.3f' % (
    mean_score, mean_precision, mean_specificity, mean_sensitivity, mean_recall))
print('Mean Balanced Accuracy: %.3f, F1 Score: %.3f' % (mean_balanced_accuracy, mean_f1))
print('Mean Confusion Matrix:\n', mean_confusion)
print('\n\nCross-Validation Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# Using Test & Train Dataset

# In[3]:


d=pd.read_csv("TestDataset.csv")
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
x_test=d.iloc[:,:-1]
y_test=d.iloc[:,-1]


# In[4]:


from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

strtfdKFold = StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X, Y)

scores = []
precisions = []
specificities = []
sensitivities = []
recalls = []
balanced_accuracies = []
f1_scores = []
confusion_matrices = []

model = RandomForestClassifier(max_depth=2, random_state=0)

for k, (train, test) in enumerate(kfold):
    model.fit(X.iloc[train], Y.iloc[train])
    y_pred = model.predict(X.iloc[test])
    score = accuracy_score(Y.iloc[test], y_pred)
    precision = precision_score(Y.iloc[test], y_pred)
    recall = recall_score(Y.iloc[test], y_pred)
    balanced_accuracy = balanced_accuracy_score(Y.iloc[test], y_pred)
    f1 = f1_score(Y.iloc[test], y_pred)
    confusion = confusion_matrix(Y.iloc[test], y_pred)

    scores.append(score)
    precisions.append(precision)
    recalls.append(recall)
    balanced_accuracies.append(balanced_accuracy)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)

    tn, fp, fn, tp = confusion.ravel()
    specificity = tn / (tn + fp)
    specificities.append(specificity)
    sensitivity = tp / (tp + fn)
    sensitivities.append(sensitivity)

    print('Fold: %2d, Training/Test Split Distribution: %s' % (k + 1, np.bincount(Y.iloc[train])))
    print('Accuracy: %.3f, Precision: %.3f, Specificity: %.3f, Sensitivity: %.3f, Recall: %.3f' % (
    score, precision, specificity, sensitivity, recall))
    print('Balanced Accuracy: %.3f, F1 Score: %.3f' % (balanced_accuracy, f1))
    print('Confusion Matrix:\n', confusion)
    print('\n')

print('\n\nCross-Validation Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
mean_score = np.mean(scores)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_balanced_accuracy = np.mean(balanced_accuracies)
mean_f1 = np.mean(f1_scores)
mean_specificity = np.mean(specificities)
mean_sensitivity = np.mean(sensitivities)
mean_confusion = np.mean(confusion_matrices, axis=0)

print('Mean Accuracy: %.3f, Precision: %.3f, Specificity: %.3f, Sensitivity: %.3f, Recall: %.3f' % (
    mean_score, mean_precision, mean_specificity, mean_sensitivity, mean_recall))
print('Mean Balanced Accuracy: %.3f, F1 Score: %.3f' % (mean_balanced_accuracy, mean_f1))
print('Mean Confusion Matrix:\n', mean_confusion)
print('\n\nCross-Validation Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[ ]:




