from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
import pandas as pd
random_state = 42

def change_data_with_selected_features(X, features, boruta=False):
    column_names=X.columns
    if boruta:
        features=np.where(features>0)
        features=features[0].tolist()
    column_names=column_names[features]
    X=X[column_names]
    return X

def change_all_datasets(train_X, val_X, test_data, selected_features, boruta=False):
    train_X_changed=change_data_with_selected_features(train_X, selected_features, boruta)
    val_X_changed=change_data_with_selected_features(val_X, selected_features, boruta)
    test_data_changed=change_data_with_selected_features(test_data, selected_features, boruta)
    return (train_X_changed, val_X_changed, test_data_changed)

def boruta_select_features(X, y):
    rf = RandomForestClassifier(max_depth=5, random_state=random_state)
    rf.fit(X, y)
    features_selection = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=random_state)
    np_y = np.array(y)
    np_y = np_y.reshape(np_y.shape[0],)
    np_X = np.array(X)
    features_selection.fit(np_X, np_y)
    features_imp_boruta_1=features_selection.support_
    features_imp_boruta_1=features_imp_boruta_1.astype(int)
    return features_imp_boruta_1

def chi2_select_features(X,y, num_feats):
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X, y)
    chi_selected_features=chi_selector.get_support()
    return chi_selected_features

def rfe_select_features(X, y, num_feats):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_selected_features = rfe_selector.get_support()
    return rfe_selected_features

def gini_select_features(X, y, if_less=False, num_feats=10,depth=5):
    rf_standard=RandomForestClassifier(max_depth=depth, random_state=random_state)
    rf_standard.fit(X, y)
    gini = rf_standard.feature_importances_
    if if_less:
        column_numbers=np.argsort(gini)[-num_feats:]
    else:
        column_numbers=np.where(gini>0)
    return column_numbers


def ss_select_features(X, y, num_feats, model, direction):
    ss = SequentialFeatureSelector(model, n_features_to_select=num_feats, direction=direction)
    ss.fit(X, y)
    ss_selected_features = ss.get_support()
    return ss_selected_features 

def l1_select_features(X, y, C=0.01):
    lsvc = LinearSVC(C=C, penalty="l1", dual=False)
    selector = SelectFromModel(lsvc).fit(X, y)
    l1_selected_features = selector.get_support()
    return l1_selected_features

def lasso_select_features(X, y, ):
    lasso = Lasso()
    SelectFromModel
    return lasso_selected_features
    
    
