
from numpy.random import seed
seed(10)
import numpy as np

import os  
#joblib用于模型保存和调用
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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold #for K-fold cross validation
import distance


def cos_sim(v1, v2):
    '''
    return cos_sim(float)
    '''
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    dot_products = np.sum(v1 * v2)
    return dot_products / (norm1 * norm2)


def get_document_dict(tok2idx, lines):
    '''
    process:计算每个词出现的文档数目，用于后续计算每个词的idf
    return :{word:exists_file_nums}
    '''
    wordco = {}
    for k, v in tok2idx.items():
        for line in lines:
            if k in line:
                if k in wordco:
                    wordco[k] = wordco[k] + 1
                else:
                    wordco[k] = 1
    return wordco

# 计算每个词的idf
def idf(wordco, leng):
    idf_all = {}
    for k, v in wordco.items():
        idf_all[k] = np.log((1.0*leng)/(1.0 * v))
    return idf_all

# 得到句子的ngram表示
# 单词型n-gram拆分
def get_ngram(sen, gram):
    gram_sen = []
    sen = sen.split()
    for i, w in enumerate(sen):
        s = ''
        for j in range(gram):
            s += sen[i+j] + ' '
        s = s.strip()
        gram_sen.append(s)
        if len(sen)-gram == i:
            break

    return gram_sen

#字符型n-gram拆分
def get_ch_ngram(sen, gram):
    gram_sen = []

    for i, w in enumerate(sen):
        s = ''
        for j in range(gram):
            s += sen[i+j] + ' '
        s = s.strip()
        gram_sen.append(s)
        if len(sen)-gram == i:
            break

    return gram_sen

# 创建词典
def creat_dic(sen, gram, tok2indx, indx2tok):
    gram_sen = get_ngram(sen, gram)
    for i, w in enumerate(gram_sen):
        if w not in tok2indx:
            #token_to_index, for words to update index
            tok2indx[w] = len(tok2indx) + 1
            indx2tok[len(tok2indx)] = w

    return tok2indx, indx2tok

# 得到训练集句子对的字典  
def load_data(path):
    '''
    return: 全局化的tok2indx和indx2tok
    '''
    tok2indx = {}
    indx2tok = {}
    if not os.path.exists(path):
        print('文件不存在，检查文件目录！')
        exit()
    text = open(path)
    lines = text.readlines()
    text.close()
    for l in lines:
        l = l.strip().split('\t')
        #句1
        sen1 = l[0]
        #句2
        sen2 = l[1]
        tok2indx, indx2tok = creat_dic(sen1, 1, tok2indx, indx2tok)
        tok2indx, indx2tok = creat_dic(sen2, 1, tok2indx, indx2tok)
    
    return tok2indx, indx2tok


def get_gram_feature(sen, tok2indx):
    '''
    params: sen:single sentence
            tok2indx: {token:index}
    process: 将一个sentence gram化，对gram化的单词列表进行token2indx映射
    return： gram_feature [idx1,idx2...]
    '''
    gram_feature = []
    gram_sen = get_ngram(sen, 1)
    for i, word in enumerate(gram_sen):
        if word in tok2indx:
            gram_feature.append(tok2indx[word])
        else:
            gram_feature.append(0)

    return gram_feature

def get_key(dic, idx):
    '''
    已知字典的value,得到key值
    '''
    for key, val in dic.items():
        if val == idx:
            return key
    return None

# 导入预训练的词embedding，句子的embedding是词的embedding和词的idf相乘之后相加取平均，句子对的表示是两个句子的分别的表示cancat，
# 在拼接两个句子表示的embedding相减得结果; 得到text_features
def get_embedding():
    '''
    result: 1.保存所有分词的idf，path=idf_all.pkl

    '''
    textfile = 'Med_corpus/Med-train.txt'
    f1 = open(textfile)
    texts = f1.readlines()
    f1.close()
    
    tok2indx = {}
    indx2tok = {}

    tok2indx, indx2tok = load_data(textfile)
    word_all = get_document_dict(tok2indx,texts)
    idf_all = idf(word_all, 1200)
    # process(tok2indx, indx2tok, textfile, grades)
    joblib.dump(idf_all, "pkl/idf_all.pkl", compress=3) 


    train_text1 = np.load('data/train_x1.npy')
    train_text2 = np.load('data/train_x2.npy')
    train_grade = np.load('data/train_y.npy')
    word_emb = np.load('data/embedding_matrix.npy')
    tokenizer = np.load('pkl/Tokenizer.pkl')
    word_dict = tokenizer.word_index

    train_text_features = np.zeros((len(train_text1), 900), "float32")
    train_text1_f = np.zeros((len(train_text1), 300), "float32")
    train_text2_f = np.zeros((len(train_text1), 300), "float32")

    for i in range(len(train_text1)):
        f1 = np.zeros((1, 300), "float32")
        f2 = np.zeros((1, 300), "float32")
        for j, indx in enumerate(train_text1[i]):
            #key is a word
            k = get_key(word_dict, indx)
            if k not in idf_all:
                idf_all[k] = 0
            f1 = f1 + word_emb[indx] * idf_all[k]
        f1 = np.divide(f1, len(train_text1[i]))
        

        for j, indx in enumerate(train_text2[i]):
            k = get_key(word_dict, indx)
            if k not in idf_all:
                idf_all[k] = 0
            f2 = f2 + word_emb[indx] * idf_all[k]
        f2 = np.divide(f2, len(train_text2[i]))
        
        # print(f1)
        # print(f2)

        f_div = f1-f2
        f_and_f1 = np.concatenate((f_div,f1),axis=1)
        train_text_features[i] = np.concatenate((f_and_f1,f2),axis=1)
    # 对特征再规范化
    text_features_scaled = preprocessing.scale(train_text_features)
    joblib.dump((text_features_scaled, train_grade), "pkl/train_feature_label.pkl", compress=3) 


def get_test_embedding():
    idf_all = joblib.load("pkl/idf_all.pkl")

    test_text1 = np.load('data/test_x1.npy')
    test_text2 = np.load('data/test_x2.npy')
    test_grade = np.load('data/test_y.npy')
    word_emb = np.load('data/embedding_matrix.npy')
    tokenizer = np.load('pkl/Tokenizer.pkl')
    word_dict = tokenizer.word_index

    test_text_features = np.zeros((len(test_text1), 900), "float32")
    test_text1_f = np.zeros((len(test_text1), 300), "float32")
    test_text2_f = np.zeros((len(test_text1), 300), "float32")

    for i in range(len(test_text1)):
        f1 = np.zeros((1, 300), "float32")
        f2 = np.zeros((1, 300), "float32")
        for j, indx in enumerate(test_text1[i]):
            k = get_key(word_dict, indx)
            if k not in idf_all:
                idf_all[k] = 0
            f1 = f1 + word_emb[indx] * idf_all[k]
        f1 = np.divide(f1, len(test_text1[i]))
        

        for j, indx in enumerate(test_text2[i]):
            k = get_key(word_dict, indx)
            if k not in idf_all:
                idf_all[k] = 0
            f2 = f2 + word_emb[indx] * idf_all[k]
        f2 = np.divide(f2, len(test_text2[i]))
        
        # print(f1)
        # print(f2)

        f_div = f1-f2
        f_and_f1 = np.concatenate((f_div,f1),axis=1)
        test_text_features[i] = np.concatenate((f_and_f1,f2),axis=1)
    # 对特征再规范化
    text_features_scaled = preprocessing.scale(test_text_features)
    joblib.dump((text_features_scaled, test_grade), "pkl/test_feature_label.pkl", compress=3) 

# 应用向量空间模型的表示进行操作得到句子对的特征(用cos_sim计算)，并把特征存储
def process(tok2indx, indx2tok, textfile, grades):
    '''
    params: tok2indx:{分词：index}
            indx2tok:{index :分词}
            textfile: 文本路径path
            grades: 标签文件路径path
    result: text_features,保存在feature_ngram.pkl中
    '''
    sen_max_len = len(tok2indx) + 1
    # print(len(tok2indx) + 1)

    f1 = open(textfile)
    f2 = open(grades)

    text = f1.readlines()
    grade = f2.readlines()

    f1.close()
    f2.close()

    all_words = get_document_dict(tok2indx, text)
    # print(all_words)
    idf_all = idf(all_words, 1500)
    # print(idf_all)
    text_features = np.zeros((len(text), 2 * sen_max_len), "float32")
    text1_f = np.zeros((len(text), len(tok2indx) + 1), "float32")
    text2_f = np.zeros((len(text), len(tok2indx) + 1), "float32")
    labels = np.zeros((len(text),1), 'float32')
    # print(tok2indx)
    # print(indx2tok)
    for i, line in enumerate(text):
        text[i] = text[i].strip().split('\t')
        f1 = get_gram_feature(text[i][0], tok2indx)
        for j, idx in enumerate(f1):
            #对于每一个单词(idx)，获取idf
            text1_f[i][idx] = idf_all[indx2tok[idx]]
        f2 = get_gram_feature(text[i][1], tok2indx)
        for j, idx in enumerate(f2):
            text2_f[i][idx] = idf_all[indx2tok[idx]]
        labels[i] = grade[i].strip()

        text1_f[i] = preprocessing.scale(text1_f[i])
        text2_f[i] = preprocessing.scale(text2_f[i])
        
        text_features[i] = cos_sim(text1_f[i], text2_f[i])

    joblib.dump(text_features, "pkl/feature_ngram.pkl", compress=3) 


  
# def train_voting(train_data, train_tags, x_test): 

#     clf1 = LogisticRegression(random_state=1)
#     clf2 = RandomForestClassifier(random_state=1)
#     clf3 = GaussianNB()
#     clf4 = KNeighborsClassifier(4)
#     clf5 = AdaBoostClassifier()
#     clf6 = SVC(kernel="linear", C=1100)
#     eclf = VotingClassifier(estimators=[
#('lr', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4), ('svm', clf6), ('adb', clf5)], voting='hard', weights=None)
#     eclf.fit(train_data, train_tags)
#     pred = eclf.predict(x_test)
#     return pred



# 算两句话之间的word level n-gram重合分数
def ngram_overlap(sen1, sen2, gram):
    gram_sen1 = get_ngram(sen1, gram)
    gram_sen2 = get_ngram(sen2, gram)
    # 去重
    new_gram_sen1 = []
    new_gram_sen2 = []
    for i, word in enumerate(gram_sen1):
        if word not in new_gram_sen1:
            new_gram_sen1.append(word)

    for i, word in enumerate(gram_sen2):
        if word not in new_gram_sen2:
            new_gram_sen2.append(word)

    l_s1 = len(new_gram_sen1)
    l_s2 = len(new_gram_sen2)
    s1_s2 = 0
    for i in range(l_s1):
        for j in range(l_s2):
            if new_gram_sen1[i]==new_gram_sen2[j]:
                s1_s2 += 1

    co = 1.0 * s1_s2 / (1.0 * (l_s1 + l_s2))
    return 2 * co

# 算两句话之间的character level n-gram重合分数
def ch_ngram_overlap(sen1, sen2, gram):
    gram_sen1 = get_ch_ngram(sen1, gram)
    gram_sen2 = get_ch_ngram(sen2, gram)
    # 去重
    new_gram_sen1 = []
    new_gram_sen2 = []
    for i, word in enumerate(gram_sen1):
        if word not in new_gram_sen1:
            new_gram_sen1.append(word)

    for i, word in enumerate(gram_sen2):
        if word not in new_gram_sen2:
            new_gram_sen2.append(word)

    l_s1 = len(new_gram_sen1)
    l_s2 = len(new_gram_sen2)
    s1_s2 = 0
    for i in range(l_s1):
        for j in range(l_s2):
            if new_gram_sen1[i]==new_gram_sen2[j]:
                s1_s2 += 1

    co = 1.0 * s1_s2 / (1.0 * (l_s1 + l_s2))
    return 2 * co


def get_gram_overlap_feature(filetext,train=None,test=None):
    '''
    process: 把每个句子对的n-gram重合的特征存储起来,以及句子长度，编辑距离特征
    params: filetext:文档
    result: sequence_feature.pkl,  sequence特征文档
            w123_gram_overlap.pkl, N-gram(词)重合特征文档 
            c2345_gram_overlap.pkl, N-gram（字符）重合特征文档 [[edit(s1,s2)]...]
            len_feature.pkl, 长度特征文档 [[len(s1),len(s2)]...]
    '''
    f = open(filetext)
    lines = f.readlines()
    f.close()
    word_gram_123_feature = []
    ch_gram_2345_features = []
    sequence_features = []
    len_feature = []

    for line in lines:
        v1 = []
        v2 = []
        v3 = []
        v4 = []
        sen1 = line.split('\t')[0]
        sen2 = line.split('\t')[1]
        v1.append(ngram_overlap(sen1, sen2, 1))
        v1.append(ngram_overlap(sen1, sen2, 2))
        v1.append(ngram_overlap(sen1, sen2, 3))
        v2.append(ch_ngram_overlap(sen1, sen2, 2))
        v2.append(ch_ngram_overlap(sen1, sen2, 3))
        v2.append(ch_ngram_overlap(sen1, sen2, 4))
        v2.append(ch_ngram_overlap(sen1, sen2, 5))
        v3.append(lc_prefix(sen1, sen2))
        v3.append(lc_suffix(sen1, sen2))
        v3.append(lc_substring(sen1, sen2))
        v3.append(lc_sequence(sen1, sen2))
        v3.append(edit_dis(sen1, sen2))
        v4.append(len(sen1))
        v4.append(len(sen2))
        word_gram_123_feature.append(v1)
        ch_gram_2345_features.append(v2)
        sequence_features.append(v3)
        len_feature.append(v4)

    if train is not None:
        joblib.dump(sequence_features, "pkl/{}sequence_feature.pkl".format('train_'), compress=3)
        joblib.dump(word_gram_123_feature, "pkl/{}w123_gram_overlap.pkl".format("train_"), compress=3)
        joblib.dump(ch_gram_2345_features, "pkl/{}c2345_gram_overlap.pkl".format("train_"), compress=3)
        joblib.dump(len_feature, "pkl/{}len_feature.pkl".format("train_"), compress=3)

    if test is not None:
        joblib.dump(sequence_features, "pkl/{}sequence_feature.pkl".format('test_'), compress=3)
        joblib.dump(word_gram_123_feature, "pkl/{}w123_gram_overlap.pkl".format("test_"), compress=3)
        joblib.dump(ch_gram_2345_features, "pkl/{}c2345_gram_overlap.pkl".format("test_"), compress=3)
        joblib.dump(len_feature, "pkl/{}len_feature.pkl".format("test_"), compress=3)

'''
获取sequence Features, 共longest common prefix/suffix/substring/sequence 
和 levenshten(edit-distance)五个features
'''
def lc_prefix(sen1, sen2):
    sen1 = sen1.split()
    sen2 = sen2.split()
    len1 = len(sen1)
    len2 = len(sen2)
    co = 0
    for i in range(len1 if len1 < len2 else len2):
        if sen1[i] == sen2[i]:
            co += 1
        else:
            break
    return co

def lc_suffix(sen1, sen2):
    sen1 = sen1.split()
    sen2 = sen2.split()
    len1 = len(sen1)
    len2 = len(sen2)
    co = 0
    for i in range(len1 if len1 < len2 else len2):
        if sen1[len1-1-i] == sen2[len2-1-i]:
            co += 1
        else:
            break
    return co

def lc_substring(sen1, sen2):
    sen1 = sen1.split()
    sen2 = sen2.split()
    len1 = len(sen1)
    len2 = len(sen2)
    co = 0
    for i in range(len1):
        for j in range(len2):
            k = 0
            while (i + k < len1 and j + k < len2):
                if sen1[i + k] != sen2[j + k]:
                    break
                k = k + 1
            co = max(co, k)
    return co

def lc_sequence(sen1, sen2):
    sen1 = sen1.split()
    sen2 = sen2.split()
    len1 = len(sen1)
    len2 = len(sen2)
    len_min = min(len1, len2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int)
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if sen1[i - 1] == sen2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
    co = dp[len1][len2]
    return co

# 计算两个句子的编辑距离
def edit_dis(s1, s2):
    return distance.levenshtein(s1, s2)

  
def evaluate(actual, pred):  
    '''
    使用pearson作为评测标准
    '''
    dev_pearson = np.corrcoef(actual, pred)[0][1]
    print('pearson:{0:.4f}'.format(dev_pearson))

def loss(y_pred, y_test):
    sum_mean=0
    for i in range(len(y_pred)):
        sum_mean+=(y_pred[i]-y_test[i])**2
    sum_erro=np.sqrt(sum_mean/150)  #这个10是你测试级的数量
    # calculate RMSE by hand
    print ("RMSE by hand:",sum_erro)
    #做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
    plt.plot(range(len(y_pred)),y_test,'r',label="test")
    plt.legend(loc="upper right") #显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()

def score_func(y_pred,y_test):
    pearson = np.corrcoef(y_pred, y_test)[0][1]
    return pearson
    
def main():
    
    # process(tok2indx, indx2tok, textfile, grades)
    # get_embedding:获取
    get_embedding()
    get_test_embedding()

    train_feature_emb, train_labels = joblib.load("pkl/train_feature_label.pkl") 
    test_feature_emb, test_labels = joblib.load("pkl/test_feature_label.pkl") 

    train_w_overlap_feature = joblib.load("pkl/train_w123_gram_overlap.pkl")
    train_c_overlap_feature = joblib.load("pkl/train_c2345_gram_overlap.pkl")

    test_w_overlap_feature = joblib.load("pkl/test_w123_gram_overlap.pkl")
    test_c_overlap_feature=joblib.load("pkl/test_c2345_gram_overlap.pkl")

    train_sequence_feature = joblib.load("pkl/train_sequence_feature.pkl")
    test_sequence_feature = joblib.load("pkl/test_sequence_feature.pkl")


    train_features = np.concatenate([train_feature_emb, train_w_overlap_feature, train_c_overlap_feature, train_sequence_feature], axis=1)
    test_features = np.concatenate([test_feature_emb, test_w_overlap_feature, test_c_overlap_feature,test_sequence_feature], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(train_features,train_labels, test_size=0.2, random_state=3)  

    

    ###########TRAIN PART###########################
    #创建评价函数
    # pearson_score=make_scorer(score_func,greater_is_better=True)
      
    # #x_train, x_test, y_train, y_test = cross_validation.tran_test_split(features,labels, test_size=0.3, random_state=3)  i

    # mean = []
    # std = []
    # kfold = KFold(n_splits=5) # k=10, split the data into 10 equal parts
    # models_name = ['RandomForestRegressor','GradientBoostingRegressor','AdaBoostRegressor']
    
    # models=[ensemble.RandomForestRegressor(n_estimators=20),
    #         ensemble.GradientBoostingRegressor(n_estimators=50),
    #         ensemble.AdaBoostRegressor(n_estimators=50)]
    # counter = 0
    # for i in models:
    #     model = i
    #     cv_result = cross_val_score(model,features,labels, cv = kfold,scoring = pearson_score)
    #     print('=============================================')
    #     print(models_name[counter])
    #     print(cv_result)
    #     print('mean: ',cv_result.mean())
    #     print('std: ',cv_result.std())
    #     counter += 1
    #     mean.append(cv_result.mean())
    #     std.append(cv_result.std())


    rf =ensemble.RandomForestRegressor(n_estimators=20)
    ada = ensemble.AdaBoostRegressor(n_estimators=50)
    gbr = ensemble.GradientBoostingRegressor(n_estimators=50)
    from xgboost import XGBRegressor
    xgb = XGBRegressor(nthread=4)


    model0=rf.fit(x_train, y_train)
    model1=gbr.fit(x_train, y_train)
    model2=ada.fit(x_train, y_train)
    model3=xgb.fit(x_train, y_train)

    #for validation part:
    print("\nIn validation:")
    y_pred0_val = model0.predict(x_val)
    print("==========for model:RandomForestRegressor==========")
    evaluate(y_pred0_val, y_val)
    y_pred1_val = model1.predict(x_val)
    print("==========for model:AdaBoostRegressor==========")
    evaluate(y_pred1_val, y_val)
    y_pred2_val = model2.predict(x_val)
    print("==========for model:GradientBoostingRegressor==========")
    evaluate(y_pred2_val, y_val)
    y_pred3_val = model3.predict(x_val)
    print("==========for model:XGBRegressor==========")
    evaluate(y_pred3_val, y_val)

    y_pred_val = np.divide((y_pred0_val+y_pred1_val+y_pred2_val+y_pred3_val),4)
    print("==========for embedding_ML_model==========")
    evaluate(y_pred_val, y_val)


    #for test part:
    print("\nIn test:")
    y_pred0 = model0.predict(test_features)
    print("==========for model:RandomForestRegressor==========")
    evaluate(y_pred0, test_labels)
    y_pred1 = model1.predict(test_features)
    print("==========for model:AdaBoostRegressor==========")
    evaluate(y_pred1, test_labels)
    y_pred2 = model2.predict(test_features)
    print("==========for model:GradientBoostingRegressor==========")
    evaluate(y_pred2, test_labels)
    y_pred3 = model3.predict(test_features)
    print("==========for model:XGBRegressor==========")
    evaluate(y_pred3, test_labels)

    y_pred = np.divide((y_pred0+y_pred1+y_pred2+y_pred3),4)
    print("==========for embedding_ML_model==========")
    evaluate(y_pred, test_labels)
    
    loss(y_pred_val, y_val)

if __name__ == '__main__':
    # main()
    # get_embedding()
    get_gram_overlap_feature('data/new_train.txt',train=True)
    get_gram_overlap_feature('data/new_test.txt',test=True)
    main()

    # print(longest_suffix("are right", "rightd"))