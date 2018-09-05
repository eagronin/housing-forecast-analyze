# Analysis

## Overview
This section describes the model for predicting home values in Denver, CO.  The model was estimated using a random forest fitted to the training data and evaluated using test data.  The model produced the test set R-squared of 0.91. 

Cleaning and processing of the data is described in the [previous section](https://eagronin.github.io/housing-forecast-prepare/).

The results are reported in the [next section](https://eagronin.github.io/housing-forecast-analyze/).

The analysis for this project was performed in Python.

## Methodology in Assessing the Data 
In order to predict home values I fitted a linear regression, the lasso and a random forest.  The lasso model was used in order to determine the most relevant features by shrinking the coefficients of less relevant features to zero.   Following the implementation of the lasso, the features with non-zero coefficients were used for fitting a random forest.   

The table provides a preliminary look at which features are likely to be important predictors of home value by showing correlations of each feature with estimated_value, lastSaleAmount and priorSaleAmount:

```
                             estimated_value  lastSaleAmount  priorSaleAmount
estimated_value                         1.00            0.79             0.62
lastSaleAmount                          0.79            1.00             0.77
priorSaleAmount                         0.62            0.77             1.00
latitude                               -0.27           -0.25            -0.23
longitude                               0.12            0.10             0.08
bedrooms                                0.37            0.28             0.24
bathrooms                               0.72            0.57             0.45
rooms                                   0.58            0.46             0.39
squareFootage                           0.82            0.65             0.54
lotSize                                 0.46            0.39             0.34
yearBuilt                               0.17            0.14             0.11
priorSaleDummy                          0.03            0.04            -0.01
rebuiltDummy                            0.18            0.06             0.06
yearsBetweenSales                       0.04            0.03             0.01
annAppreciation                        -0.03            0.01            -0.17
Dummy2012ForLastSaleAmount              0.01            0.03            -0.02
lastSaleAmountAfter2012                 0.38            0.47             0.33
Dummy2012ForPriorSaleAmount            -0.02            0.01            -0.03
priorSaleAmountAfter2012                0.16            0.36             0.29
80203                                   0.01            0.02             0.02
80204                                  -0.23           -0.20            -0.16
80205                                  -0.15           -0.15            -0.14
80206                                   0.32            0.27             0.23
80207                                  -0.12           -0.09            -0.08
80209                                   0.27            0.24             0.21
80123                                  -0.04           -0.01             0.01
```

As a first step, I split the dataset into training and test sets.  Each model was fitted using the training set and evaluated using the test set.  The implementation is shown below:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics.regression import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from comps_features import comps_features


def train_test_evaluation(data, use_comps, non_linear, rseed, n_est=10, max_d=None, min_samp_split=2, min_samp_leaf=1, max_f='auto'):

    y = data.estimated_value
    X = data    # keep zestimate (estimated_value) in the data for now and drop it later
        
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rseed)
    y_train = X_train.estimated_value    
    y_test = X_test.estimated_value
    
    # add home valuation features based on comparables
    if use_comps == 1:
        # create a dataset from which comparables for each home will be selected
        train = X_train[['id', 'latitude', 'longitude', 'estimated_value', 'lastSaleAmount', 'lastSaleDate', 'squareFootage', 'bedrooms', 'bathrooms']]
        X_train = comps_features(X_train, train)
        X_test = comps_features(X_test, train)
    
    # remove features that should not be used as independent features in fitting the model
    X_train.drop(['estimated_value', 'id', 'lastSaleDate', 'priorSaleDate', 'zipcode'], axis = 1, inplace = True)
    X_test.drop(['estimated_value', 'id', 'lastSaleDate', 'priorSaleDate', 'zipcode'], axis = 1, inplace = True)
            
    # evaluate and select model using training and test sets
    X_columns = X_test.columns
    
    # add non-linearities
    if non_linear == 1:
        poly = PolynomialFeatures(degree = 2)
        X_train = poly.fit_transform(X_train)
        X_test = poly.fit_transform(X_test)        
        #z = pd.DataFrame(X_train, columns = poly.get_feature_names(X_columns))
        #z = z[['lastSaleAmount', 'priorSaleAmount', 'compVal']]
        #z = z.reset_index(drop = True)
        #y_train = y_train.reset_index(drop = True)
        #z = pd.concat([z, y_train], axis = 1)
        #z.corr()
   
    # linear regression
    lm = LinearRegression().fit(X_train, y_train)
    y_pred_train = lm.predict(X_train)
    R2_lm_train = r2_score(y_train, y_pred_train)
    
    y_pred_test = lm.predict(X_test)
    R2_lm_test = r2_score(y_test, y_pred_test)

    # lasso
    scaler = StandardScaler().fit(X_train)
    X_train_st = scaler.transform(X_train)
    X_test_st = scaler.transform(X_test)
    
    ls = Lasso(alpha = 100, max_iter = 10000, normalize = True, random_state = rseed).fit(X_train_st, y_train)
    y_pred_train = ls.predict(X_train_st)
    R2_ls_train = r2_score(y_train, y_pred_train)
    
    y_pred_test = ls.predict(X_test_st)
    R2_ls_test = r2_score(y_test, y_pred_test)
    
    if non_linear == 1:
        # for a model with poynomial features:
        out = pd.DataFrame(poly.get_feature_names(X_columns), ls.coef_, columns = ['variables'])
    else:        
        # for a model without polynomial features:
        out = pd.DataFrame(X_columns, ls.coef_, columns = ['variables'])
    
    # keep features with non-zero coefficients only:
    out = out.reset_index(drop = False)
    out = out.rename(columns = {'index': 'coefficients'})
    out = out[['variables', 'coefficients']]
    relFeatures = out[abs(out.coefficients) > 0.0]

    # random forest
    X_train_rf = pd.DataFrame(X_train, columns = out.variables.tolist())
    X_test_rf = pd.DataFrame(X_test, columns = out.variables.tolist())    
    # if we want to keep only the features selected using regularization:
    X_train_rf = X_train_rf[relFeatures.variables.tolist()]
    X_test_rf = X_test_rf[relFeatures.variables.tolist()]
    
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_split=min_samp_split, min_samples_leaf=min_samp_leaf, max_features=max_f, random_state = rseed).fit(X_train_rf, y_train)
    y_pred_train = rf.predict(X_train_rf)
    R2_rf_train = r2_score(y_train, y_pred_train)
    
    y_pred_test = rf.predict(X_test_rf)
    R2_rf_test = r2_score(y_test, y_pred_test)
    
    featureImportances = pd.DataFrame(X_test_rf.columns, rf.feature_importances_)    
    featureImportances = featureImportances.sort_index(ascending = False)
    featureImportances = featureImportances.rename(columns = {0: 'featureImportances'})
    featureImportances = featureImportances.reset_index(drop = False)
        
    return R2_lm_train, R2_lm_test, R2_ls_train, R2_ls_test, R2_rf_train, R2_rf_test, relFeatures, featureImportances
```

Next step:  [Results](https://eagronin.github.io/housing-forecast-report/).
