import pandas as pd
import matplotlib.pyplot as plt
from source.feature_selection_methods import change_data_with_selected_features, boruta_select_features, chi2_select_features, rfe_select_features, gini_select_features, ss_select_features, l1_select_features
from source.models import train_model

def  plot_results2(list_features, list_scores, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle(title)
    for i in range(len(list_features)):
        ax1.plot(list_scores[i]['BA'], label=list_features[i])
        ax2.plot(list_scores[i]['Score'], label=list_features[i])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('BA')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def  plot_results(end_scores, title):
    models=['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle(title)
    for i in range(len(models)):
        model_scores = end_scores[end_scores['model']==models[i]]
        ax1.plot(model_scores['features'], model_scores['BA'], label=models[i])
        ax2.plot(model_scores['features'], model_scores['Score'], label=models[i])
    ax1.set_xlabel('Features')
    ax1.set_ylabel('BA')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def search_best_features2(num_possible_features, method, train_X, train_y, val_X, val_y, dataset_name):
    models=['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
    index_list=[]
    for num_features in num_possible_features:
        for model in models:
            index_list.append(f'{model} {num_features}')
    all_scores_=pd.DataFrame(index=index_list, columns=['BA', 'Score'])
    end_scores=[]
    for i in range(len(num_possible_features)):
        if method=='chi2':
            features = chi2_select_features(train_X, train_y, num_possible_features[i])
        elif method=='rfe':
            features = rfe_select_features(train_X, train_y, num_possible_features[i])
        elif method=='gini':
            features = gini_select_features(train_X, train_y, num_possible_features[i])
        else:
            print('Wrong method')
            return
        train_X_after=change_data_with_selected_features(train_X, features)
        val_X_after=change_data_with_selected_features(val_X, features)

        for model in models:
            BA, score=train_model(model, train_X_after, train_y, val_X_after, val_y, dataset_name)
            all_scores_.loc[str(model+" "+str(num_possible_features[i]))]=[BA, score]

        to_append=all_scores_.iloc[i*7:(i+1)*7]
        to_append.index = ['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
        end_scores.append(to_append)
    plot_results(num_possible_features, end_scores, str(method)+" "+str(dataset_name))

def search_best_features(num_possible_features, method, train_X, train_y, val_X, val_y, dataset_name):
    models=['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
    index_list=[]
    for model in models:
        for num_features in num_possible_features:
            index_list.append(f'{model} {num_features}')
    end_scores=pd.DataFrame(columns=['model', 'features', 'BA', 'Score'])
    
    for model in models:
        for i in range(len(num_possible_features)):
            if method=='chi2':
                features = chi2_select_features(train_X, train_y, num_possible_features[i])
            elif method=='rfe':
                features = rfe_select_features(train_X, train_y, num_possible_features[i])
            elif method=='gini':
                features = gini_select_features(train_X, train_y, num_possible_features[i])
            elif method=="sfs":
                features = ss_select_features(train_X, train_y, num_possible_features[i], model, direction="forward")
            elif method == "sbs":
                features = ss_select_features(train_X, train_y, num_possible_features[i], model, direction="forward")
            else:
                print('Wrong method')
                return
            train_X_after=change_data_with_selected_features(train_X, features)
            val_X_after=change_data_with_selected_features(val_X, features)

            BA, score=train_model(model, train_X_after, train_y, val_X_after, val_y, dataset_name)
            end_scores.loc[len(end_scores)] = [model, num_possible_features[i], BA, score]
    return(end_scores)


def search_best_C(possible_Cs, train_X, train_y, val_X, val_y, dataset_name):
    models=['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
    index_list=[]
    for model in models:
        for C in possible_Cs:
            index_list.append(f'{model} {C}')
    end_scores=pd.DataFrame(columns=['model', 'C', 'BA', 'Score'])
    
    for model in models:
        for i in range(len(possible_Cs)):
            features = l1_select_features(train_X, train_y, C=possible_Cs[i])
            
            train_X_after=change_data_with_selected_features(train_X, features)
            val_X_after=change_data_with_selected_features(val_X, features)

            BA, score=train_model(model, train_X_after, train_y, val_X_after, val_y, dataset_name)
            end_scores.loc[len(end_scores)] = [model, possible_Cs[i], BA, score]
    return(end_scores)

def  plot_results_best_C(end_scores, title):
    models=['xgb', 'lda', 'svc', 'lr', 'rf', 'dt', 'knn']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle(title)
    for i in range(len(models)):
        model_scores = end_scores[end_scores['model']==models[i]]
        ax1.plot(model_scores['C'], model_scores['BA'], label=models[i])
        ax2.plot(model_scores['C'], model_scores['Score'], label=models[i])
    ax1.set_xlabel('C')
    ax1.set_ylabel('BA')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax2.set_xlabel('C')
    ax2.set_ylabel('Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

