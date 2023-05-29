import pandas as pd
random_state = 42
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def read_artificial_data(dir_path="."):
    art_train_data = pd.read_csv(dir_path+'/artificial_train.data', sep=' ', header=None)
    art_train_data = art_train_data.drop(art_train_data.columns[500], axis=1)
    art_train_labels = pd.read_csv(dir_path+'/artificial_train.labels', sep=' ', header=None)
    art_train_labels[art_train_labels == -1] = 0
    art_train_X, art_val_X, art_train_y, art_val_y = train_test_split(art_train_data, art_train_labels, test_size=0.2, random_state=42)
    art_test_data = pd.read_csv(dir_path+'/artificial_valid.data', sep=' ', header=None)
    art_test_data = art_test_data.drop(art_test_data.columns[500], axis=1)
    return art_train_X, art_train_y, art_val_X, art_val_y, art_test_data


def read_spam(dir_path='.'):
    spam_train = pd.read_csv(dir_path+'/sms_train.csv', sep=',')
    spam_test_data = pd.read_csv(dir_path+'/sms_test.csv', sep=',')
    spam_train_data, spam_train_labels = spam_train['message'], spam_train['label']
    
    spam_train_X, spam_val_X, spam_train_y, spam_val_y = train_test_split(spam_train_data, spam_train_labels, test_size=0.2, random_state=random_state)

    tfidf = TfidfVectorizer()
    train = tfidf.fit_transform(spam_train_X)
    spam_train_X = pd.DataFrame(train.toarray())

    val = tfidf.transform(spam_val_X)
    spam_val_X = pd.DataFrame(val.toarray())

    test = tfidf.transform(spam_test_data['message'])
    spam_test_data = pd.DataFrame(test.toarray())

    minmax=MinMaxScaler()
    spam_train_X=pd.DataFrame(minmax.fit_transform(spam_train_X), columns=tfidf.get_feature_names_out())
    spam_val_X=pd.DataFrame(minmax.transform(spam_val_X), columns=tfidf.get_feature_names_out())
    spam_test_data=pd.DataFrame(minmax.transform(spam_test_data), columns=tfidf.get_feature_names_out())
    
    return spam_train_X, spam_train_y, spam_val_X, spam_val_y, spam_test_data