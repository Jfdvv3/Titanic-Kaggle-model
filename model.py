# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

"""Take this file and put it in a folder with the titanic datasets."""


""" Read the training data csv file"""
Titdata = pd.read_csv('train.csv')
#print(Titdata.columns)


""" Read the test data csv file"""
Testdata = pd.read_csv('test.csv')

"""Read the test data survived values"""



m = {'m' : 1, 'f' : 0}
"""Replace words male with 1 and female with 0- Jako you'll like this bit"""
Titdata['Sex'] = Titdata['Sex'].str[0].str.lower().map(m)
Testdata['Sex'] = Testdata['Sex'].str[0].str.lower().map(m)


""" Take these 4 features out- because they are probably the biggest factors
    that don't need much manipulation """
Tit_features = ['Pclass', 'Sex', 'Age', 'Fare']
X = Titdata[Tit_features]


"""There are some ages that haven't been given so i'm gonna replace them
    with 0, probably not the best solution but oh well."""
Xtrain = X.fillna(0)


"""The y column is the output values"""
ytrain = Titdata['Survived']


"""We need the same data from the test csv file"""
Xtest = Testdata[Tit_features]
Xtest = Xtest.fillna(0)
#ytest = Testdata['Survived']


"""let's make a decision tree regressor for this one"""
Titmodel = DecisionTreeClassifier(random_state=1)
Titmodel.fit(Xtrain,ytrain)
TitDT_preds = Titmodel.predict(Xtest)


"""Write it to a csv file"""
Testdata['Survived'] = pd.Series(TitDT_preds, index=Testdata.index)
finalsolutions = ['PassengerId', 'Survived']
Out = Testdata[finalsolutions]
Out.to_csv('Solutions.csv', index=False)
#print('Decision Tree model error:')
#print(mean_absolute_error(ytest, TitDT_preds))


"""I still don't fully know what a random forest is, but it turns out to be
    very easy to try, so let's give it a go too"""
forest_model = RandomForestClassifier(random_state=2)
forest_model.fit(Xtrain, ytrain)
TitFor_preds = forest_model.predict(Xtest)


"""Try a Random forest and put it into a csv file"""
Testdata['Survived'] = pd.Series(TitFor_preds, index=Testdata.index)
finalsolutions2 = ['PassengerId', 'Survived']
Out = Testdata[finalsolutions2]
Out.to_csv('SolutionsForest.csv', index=False)