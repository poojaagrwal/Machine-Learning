from sklearn import datasets, svm
import numpy as np

digits= datasets.load_digits()

x_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1, kernel = 'linear')

print("score without kfold and cross validation")
print(svc.fit(x_digits[:-100], y_digits[:-100]).score(x_digits[-100:], y_digits[-100:])) 

''' To get a better measure of prediction accuracy (which we can use as a proxy for 
goodness of fit of the model), we can successively split the data in folds that 
we use for training and testing:'''


x_folds= np.array_split(x_digits,3)
y_folds= np.array_split(y_digits,3)

scores= list()

for k in range(3):
    x_train = list(x_folds)
    
    x_test = x_train.pop(k)
   
    x_train = np.concatenate(x_train)
    
    y_train = list(y_folds)
    
    y_test = y_train.pop(k)
    
    y_train = np.concatenate(y_train)
    
    scores.append(svc.fit(x_train, y_train).score(x_test,y_test))
print("score with manual kfold")
print(scores)



from sklearn.model_selection import KFold, cross_val_score

k_fold = KFold(n_splits=3)
'''
Explaining k-fold with small example
X= ['a','a','b','c','c']
print(list(k_fold.split(X)))

for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test %s' % (train_indices, test_indices))
    
simlilarly'''

score= [svc.fit(x_digits[train],y_digits[train]).score(x_digits[test],y_digits[test]) for train,test in k_fold.split(x_digits)]  
print("score with kfold and cross validation")
print(score)   

print("score using cross_val_score")
print(cross_val_score(svc, x_digits, y_digits, cv=k_fold))
   