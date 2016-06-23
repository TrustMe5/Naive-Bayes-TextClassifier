#!/usr/bin/env python
# coding=utf-8
import os
import jieba
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.naive_bayes import MultinomialNB
def wordset(filepath):
    all_words=[]
    with open(filepath,'r') as fp:
        lines=fp.readlines()
    for line in lines:
        word=line.strip().decode('unicode_escape')
        if len(word) and word not in all_words:
            all_words.append(word)
    return all_words

def TextProcessing(train_data_folder,test_data_folder):
    train_data_list=[]
    train_class_list=[]
    train_data_folder_list=os.listdir(train_data_folder)
    for folder in train_data_folder_list:
        new_path_folder=os.path.join(train_data_folder,folder)
        files=os.listdir(new_path_folder)
        for file in files:
            new_path_file=os.path.join(new_path_folder,file)
            with open(new_path_file,'r') as fp:
                raw=fp.read()
                words=list(jieba.cut(raw,cut_all=False))
                train_data_list.append(words)
                train_class_list.append(folder.decode('utf-8'))


    test_data_list=[]
    test_data_files=os.listdir(test_data_folder)
    for file in test_data_files:
        new_path_file=os.path.join(test_data_folder,file)
        with open(new_path_file,'r') as fp:
            raw=fp.read()
            words=list(jieba.cut(raw,cut_all=False))
            test_data_list.append(words)

    all_words_list={}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_list.has_key(word):
                all_words_list[word]+=1
            else:
                all_words_list[word]=1
    all_words_tuple_list=sorted(all_words_list.items(),key=lambda f:f[1],reverse=True)
    all_words_list=list(zip(*all_words_tuple_list)[0])
    return all_words_list,train_data_list,train_class_list,test_data_list


def word_dic(all_words_list,deleteN,stopwords):
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_words_list),1):
        if n>100:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords and 0<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n+=1
    return feature_words

def TextFeatures(train_data_list,test_data_list,feature_words):
    def text_features(text,feature_words):
        text_words=set(text)
        features=[1 if word in text_words else 0 for word in feature_words]
        return features
    train_features=[text_features(text,feature_words) for text in train_data_list]
    test_features=[text_features(text,feature_words) for text in test_data_list]
    return train_features,test_features

def TextClassifify(train_features,train_class_list,test_data_list):
    classifier=MultinomialNB().fit(train_features,train_class_list)
    test_class_list=classifier.predict(test_data_list)
    return test_class_list

if __name__=='__main__':
    print 'start'

    train_data_folder='./Database/SogouC/Sample/'

    test_data_folder='./Database/SogouC/TestArticle/'

    all_words_list,train_data_list,train_class_list,test_data_list=TextProcessing(train_data_folder,test_data_folder)
    stopwords_file='./stopwords_cn.txt'
    stopwords=wordset(stopwords_file)

    deleteNs=range(0,1000,20)
    for deleteN in deleteNs:
        feature_words=word_dic(all_words_list,deleteN,stopwords)
        train_features,test_features=TextFeatures(train_data_list,test_data_list,feature_words)
        test_class_list=TextClassifify(train_features,train_class_list,test_features)
        print test_class_list


