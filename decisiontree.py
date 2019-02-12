from sklearn import tree
from subprocess import call

# Doing classification using DecisionTree Classifier
clf = tree.DecisionTreeClassifier()

#Using numeric representation of the Categorical values for matrix representation

##Outlook:   Sunny  =1, Rain=2, Overcast =3
##Temperature:   Hot = 1, Mild =2, Cool = 3
##Humidity:   High = 0, Normal= 1
##Wind:   Weak = 0, Strong = 1 

#Training Data without labels/class

X = [[1,1,0,0],
     [1,1,0,1],
     [3,1,0,0],
     [2,2,0,0],
     [2,3,1,0],
     [2,3,1,1],
     [3,3,1,1],
     [1,2,0,0],
     [1,3,1,0],
     [2,2,1,0],
     [1,2,1,1],
     [3,2,0,1],
     [3,1,1,0],
     [2,2,0,1]]

#Training Class data

Y= ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#Using fit method to feed the training data in classifier

clf.fit(X,Y)

#using graphviz to genrate dot data

tree.export_graphviz(clf, out_file='decisionTree.dot', feature_names=['Outlook','Temperature','Humidity','Wind'],
                     class_names=['No','Yes'],filled =True,rounded=True)

#Convert dot data into image(png)
call(['dot', '-T', 'png','decisionTree.dot', '-o','decisionTree.png'])

#Predicting the class of test data

#Outlook =Sunny, Temperature= Hot, Humidity = Normal, Wind = Weak
print(clf.predict([[1,1,1,0]]))