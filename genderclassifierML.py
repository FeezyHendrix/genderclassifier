from sklearn import tree, neighbors,svm, naive_bayes
from sklearn.metrics import accuracy_score

#initializing dependencies
clf = tree.DecisionTreeClassifier()
clf1 = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()
clf3 = naive_bayes.GaussianNB()

#data sets
X = [[181,80,44], [177, 70, 43], [160,60,38], [154, 54, 37],[165, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y= ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']


#train the models on data
clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
#data predictions
firstModel = clf1.predict(X)
secondModel = clf2.predict(X)
thirdModel = clf3.predict(X)

#assigning the accuracy_score

firstmodelAccuracy = {'name' : 'KNeighborsClassifier',
                      'accuracy' : accuracy_score(firstModel, Y)
                     }
secondmodelAccuracy = {'name': 'svm',
                       'accuracy' : accuracy_score(secondModel, Y)
                      }
thirdmodelAccuracy = {'name': 'naive_bayes',
                       'accuracy' : accuracy_score(thirdModel, Y)
                      }

if (firstmodelAccuracy['accuracy'] > secondmodelAccuracy['accuracy']  and firstmodelAccuracy['accuracy'] > thirdmodelAccuracy['accuracy']):
    best = firstmodelAccuracy
elif (secondmodelAccuracy['accuracy'] > firstmodelAccuracy['accuracy'] and secondmodelAccuracy['accuracy'] > thirdmodelAccuracy['accuracy']):
    best = secondmodelAccuracy
else:
    best = thirdmodelAccuracy

print(best['name'], "has the accurate prediction with and accuracy of ", best['accuracy'])
