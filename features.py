from numpy.random import seed
seed(10)
import numpy as np

import os  
from sklearn.externals import joblib   
from sklearn import preprocessing  
import math  
import sys  
from sklearn import metrics 
from sklearn.feature_extraction.text import HashingVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer   
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC  
from sklearn.svm import LinearSVC  
from sklearn.feature_selection import SelectPercentile, f_classif  
from sklearn.feature_selection import SelectKBest  
from sklearn.feature_selection import chi2  
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold #for K-fold cross validation
import xgboost as xgb
import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def cos_sim(v1, v2):
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    dot_products = np.sum(v1 * v2)
    return dot_products / (norm1 * norm2)

# 计算两个句子的编辑距离
def edit_dis(s1, s2):
    return distance.levenshtein(s1, s2)

def n-gram_features(data):
	# step 1
	vectoerizer = CountVectorizer(min_df=1e-2, max_df=1.0, ngram_range=(3,3), analyzer='char_wb')
	# step 2
	vectoerizer.fit(data)
	# step 3
	bag_of_words = vectoerizer.get_feature_names()
	print("Bag of words:")
	print(bag_of_words)
	print(len(bag_of_words))
	# step 4
	X = vectoerizer.transform(data)
	# print("Vectorized corpus:")
	# print(X.toarray())
	# step 5
	# print("index of `的` is : {}".format(vectoerizer.vocabulary_.get('的')))

	# step 1
	tfidf_transformer = TfidfTransformer()
	# step 2
	tfidf_transformer.fit(X.toarray())
	# step 3
	for idx, word in enumerate(vectoerizer.get_feature_names()):
	  print("{}\t{}".format(word, tfidf_transformer.idf_[idx]))
	# step 4
	tfidf = tfidf_transformer.transform(X)
	tfidf = tfidf.toarray()

	joblib.dump(tfidf, "embedding_features/train_tfidf_features.pkl", compress=3) 

def score_func(y_pred,y_test):
	pearson = np.corrcoef(y_pred, y_test)[0][1]
	return pearson

def xgb():
	params = {
			colsample_bytree=0.4603, 
			gamma=0.0468,
			learning_rate=0.05, 
			max_depth=3, 
            min_child_weight=1.7817, 
            n_estimators=50,
            reg_alpha=0.4640, 
            reg_lambda=0.8571,
            subsample=0.5213, 
            silent=1,
            random_state =7, 
            nthread = -1}
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit()
    pre = xgb_model.predict()

def train():
	#x_train, x_test, y_train, y_test = cross_validation.tran_test_split(features,labels, test_size=0.3, random_state=3)  i

    mean = []
    std = []
    kfold = KFold(n_splits=5) # k=10, split the data into 10 equal parts
    models_name = ['RandomForestRegressor','GradientBoostingRegressor','AdaBoostRegressor']
    
    models=[ensemble.RandomForestRegressor(n_estimators=20),
            ensemble.GradientBoostingRegressor(n_estimators=50),
            ensemble.AdaBoostRegressor(n_estimators=50)]
    counter = 0
    for i in models:
        model = i
        cv_result = cross_val_score(model,features,labels, cv = kfold,scoring = pearson_score)
        print('=============================================')
        print(models_name[counter])
        print(cv_result)
        print('mean: ',cv_result.mean())
        print('std: ',cv_result.std())
        counter += 1
        mean.append(cv_result.mean())
        std.append(cv_result.std())
