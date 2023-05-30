import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from source.metrics import balanced_accuracy, artificial_score, spam_score
from source.feature_selection_methods import change_data_with_selected_features, ss_select_features

def model_by_name(model_name):
    if model_name=='xgb':
        model=xgb.XGBClassifier()
    elif model_name=='lda':
        model = LinearDiscriminantAnalysis()
    elif model_name=='svc':
        model = SVC()
    elif model_name=='lr':
        model = LogisticRegression()
    elif model_name=='rf':
        model = RandomForestClassifier()
    elif model_name=='dt':
        model = DecisionTreeClassifier()
    elif model_name=='knn':
        model = KNeighborsClassifier()
    else:
        print('Wrong model name')
        return
    return model

def train_model(model_name, X_train, y_train, X_val, y_val, dataset_name):
    model = model_by_name(model_name)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_val)
    BA=balanced_accuracy(y_val, y_pred)
    if dataset_name=='artificial':
        score=artificial_score(BA, len(X_train.columns))
    elif dataset_name=='spam':
        score=spam_score(BA, len(X_train.columns))
    return BA, score

def sequential_search_train_model(model_name, X_train, y_train, X_val, y_val, dataset_name, num_feats, direction):
    model = model_by_name(model_name)
    train_X_ss_features=ss_select_features(X_train, y_train, num_feats, model, direction)
    train_X_ss=change_data_with_selected_features(X_train, train_X_ss_features)
    val_X_ss=change_data_with_selected_features(X_val, train_X_ss_features)

    model.fit(train_X_ss, y_train)
    y_pred=model.predict(val_X_ss)
    BA=balanced_accuracy(y_val, y_pred)
    if dataset_name=='artificial':
        score=artificial_score(BA, len(X_train.columns))
    elif dataset_name=='spam':
        score=spam_score(BA, len(X_train.columns))
    return BA, score