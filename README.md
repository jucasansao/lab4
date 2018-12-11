# Laboratory 4

This laboratory focuses on designing classifiers committees called ensemble. At the beginning, you will test existing implementations of the ensemble from the sklearn library. The next step will be designing your own solutions.

Below is an example of using the ensemble from the `sklearn` library.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create ensemble
AdaB = AdaBoostClassifier(base_estimator=SVC(), n_estimators=10)
# Train classifier
AdaB.fit(X_train, y_train)
# Predict test data
y_pred = AdaB.predict(X_test)
# Calculate accuracy
score = accuracy_score(y_pred, y_test)
```

Please put solutions in the [`solution.py`](solution.py) file.

## Exercises

### Exercise 1 (2 pts)

> Tip: In ensembles use [`k-NN`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) as base estimator

- Load data from a file[`phoneme.csv`](data/phoneme.csv)
- Divide the set into the feature set (`X`) and the set of labels (`y`). Use the `prepare_data` function from the `utils.py` file
- Make a simple division of the set into teaching (`X_train`,` y_train`) and test (`X_test`, `y_test`). Maintain a 30% ratio for learning and 70% for testing [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- For [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) ensembles and [`k-NN`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) classifier
  - Learn the classifier/ensemble on the training set
  - Calculate the prediction for the learned classifier on the test set
  - Put the [`accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) on the list for the classification of the tested classifiers
- Answer in comment which achieve best score

### Exercise 2 (3 pts)

> As an ensemble use [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) class. Create different combination.
> Ensembles lists:
>
> 1. ([`k-NN`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier), k-NN, k-NN)
> 2. ([`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), SVC, SVC)
> 3. ([`GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), GaussianNB, GaussianNB)
> 4. (k-NN, SVC, GaussianNB)

- Load data from a file [`balance.csv`](data/balance.csv)
- Divide the set into the feature set (`X`) and the set of labels (`y`). Use the `prepare_data` function from the `utils.py` file
- Make a k-fold cross validation for `k = 10` for the loaded data [`k-fold cross validation`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- For each fold:
  - Learn the ensemble on the training set
  - Calculate the prediction for the learned ensemble on the test set
  - Put the accuracy on the list for the classification of the training set on the given fold
- Display on the screen the average accuracy of the classification and its standard deviation
- Check difference of using hard voting and soft voting
- Answer in comment which ensemble achieve best score with which type of voting

### Exercise 3 (5pts)

- Create class with your own ensemble implementation
- This class should use at least 2 parameters - base_estimator and number_of_estimators
- Ensemble is an array of number_of_estimators * base classifiers
- Extend your ensemble with random subspace method
  - Create n different feature subsets indexes
  - Put it in f_array
  - Train each classifier with data only with selected features (f_array)
  - In predict method use only selected features (f_array)
- Implement fit and predict function
- Use majority voting technique
- Compare with classifiers from exercise 2
- Use from a file [`coil2000.csv`](data/coil2000.csv) or file [`sonar.csv`](data/sonar.csv)
