import numpy as np

def balanced_accuracy(y_true, y_pred):
    #y true - np array of true labels
    #y pred - np array of predicted labels
    true_positive=0
    true_negative=0
    positive=0
    negative=0
    y_true=np.array(y_true)
    for i in range(len(y_true)):
        if y_true[i]==1:
            positive+=1
            if y_pred[i]==1:
                true_positive+=1
        else:
            negative+=1
            if y_pred[i]==0:
                true_negative+=1
    balanced_accuracy=1/2*(true_positive/positive+true_negative/negative)
    return balanced_accuracy

def artificial_score(BA, num_features):
    score=BA-0.01*max(0, 1/5*num_features-1)
    return score

def spam_score(BA, num_features):
    score=BA-0.01*max(0, 1/100*num_features-1)
    return score
