#!/usr/bin/env python
# coding=utf-8
import os
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

#def TextProcessing(folder_path,test_size):
#    folders=os.listdir(folder_path)
#    data_list=[]
#    class_list=[]
#
#    for folder in folders:
#        new_folder_path=os.path.join(folder_path,folder)
#        files=os.listdir(new_folder_path)
#        for file in files:
#            with open(os.path.join(new_folder_path,file),'r') as fp:
#                raw=fp.read()
#            word_list=list(jieba.cut(raw,cut_all=False))
#            data_list.append(word_list)
#            class_list.append(folder.decode('utf-8'))
#
#    data_class_list=zip(data_list,class_list)
#    #print data_class_list
#    random.shuffle(data_class_list)
#    index=int(len(data_class_list)*test_size)+1
#    train_list=data_class_list[index:]
#    test_list=data_class_list[:index]
#    train_data_list,train_class_list=zip(*train_list)
#    test_data_list,test_class_list=zip(*test_list)
#
#
#    all_words_list={}
#
#    for word_list in train_data_list:
#        for word in word_list:
#            if all_words_list.has_key(word):
#                all_words_list[word]+=1
#            else:
#                all_words_list[word]=1
#
#    all_words_tuple_list=sorted(all_words_list.items(),key=lambda f:f[1],reverse=True)
#    all_words_list=list(zip(*all_words_tuple_list)[0])
#    return all_words_list,train_data_list,train_class_list,test_data_list,test_class_list



#处理训练集和待预测的文章集
def Text_Processing(train_folder_path,test_folder_path):
    train_folders=os.listdir(train_folder_path)
    train_data_list=[]
    train_class_list=[]
    for folder in train_folders:
        new_folder_path=os.path.join(train_folder_path,folder)
        files=os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path,file),'r') as fp:
                raw=fp.read()
                word_list=list(jieba.cut(raw,cut_all=False))
                train_data_list.append(word_list)
                train_class_list.append(folder.decode('utf-8'))
    all_words_list={}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_list.has_key(word):
                all_words_list[word]+=1
            else:
                all_words_list[word]=1
    all_words_tuple_list=sorted(all_words_list.items(),key=lambda f:f[1],reverse=True)
    all_words_list=list(zip(*all_words_tuple_list)[0])


    test_files=os.listdir(test_folder_path)
    test_data_list=[]
    for file in test_files:
        new_folder_path=os.path.join(test_folder_path,file)
        with open(new_folder_path,'r') as fp:
            raw=fp.read()
            word_list=list(jieba.cut(raw,cut_all=False))
            test_data_list.append(word_list)
    return all_words_list,train_data_list,train_class_list,test_data_list







def MakeWordSet(file):
    word_list=set()
    with open(file,'r') as fp:
        lines=fp.readlines()
    for line in lines:
        word=line.strip().decode('unicode_escape')
        if len(word) and word not in word_list:
            word_list.add(word)
    return word_list

def words_dic(all_words_list,deleteN,stopwords_set=set()):           #get feature_words
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_words_list),1):
        if n>100:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 0<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n+=1
    return feature_words



def TextFeatures(train_data_list,test_data_list,feature_words,flag='nltk'): 
    def text_features(text,feature_words):
        text_words=set(text)
        if flag=='nltk':
            features={word:1 if word in text_words else 0 for word in feature_words}
        elif flag=='sklearn':
            features=[1 if word in text_words else 0 for word in feature_words]
        else:
            features=[]
        return features
    train_feature_list=[text_features(text,feature_words) for text in train_data_list]
    test_feature_list=[text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list
#def TextClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag='nltk'):
#    if flag=='nltk':
#        train_flist=zip(train_feature_list,train_class_list)
#        test_flist=zip(test_feature_list,test_class_list)
#        classifier=nltk.classify.NaiveBayesClassifier.train(train_flist)
#        test_accuracy=nltk.classify.accuracy(classifier,test_flist)
#    elif flag=='sklearn':
#        classifier=MultinomialNB().fit(train_feature_list,train_class_list)
#        test_accuracy=classifier.score(test_feature_list,test_class_list)
#        test_class_list1=classifier.predict(test_feature_list)
#    else:
#        test_accuracy=[]
#    print 'test_class_list1:',test_class_list1
#    return test_accuracy
#
#进行预测
def Text_Classifier(train_feature_list,test_feature_list,train_class_list):
    classifier=MultinomialNB().fit(train_feature_list,train_class_list)
    test_class_list=classifier.predict(test_feature_list)
    return test_class_list


if __name__=='__main__':
    print 'start'

    #folder_path='./Database/SogouC/Sample/'
    train_folder_path='./Database/SogouC/Sample/'
    
    test_folder_path='./Database/SogouC/TestArticle/'

    #all_words_list,train_data_list,train_class_list,test_data_list,test_class_list=TextProcessing(folder_path,test_size=0.2)
    #print 'train_data_list:',train_data_list[0]
    #print 'all_words_list[100]:',all_words_list[100]
   # j=0
   # for i in range(len(all_words_list)):
   #     if all_words_list[i]==all_words_list[10]:
   #         j+=1
    all_words_list,train_data_list,train_class_list,test_data_list=Text_Processing(train_folder_path,test_folder_path)

    stopwords_file='stopwords_cn.txt'
    stopwords_set=MakeWordSet(stopwords_file)


    flag='sklearn'
    deleteNs=range(0,1000,20)
    #print deleteNs
#    test_accuracy_list=[]
    for deleteN in deleteNs:
        feature_words=words_dic(all_words_list,deleteN,stopwords_set)
        print 'feature_words:',
        for i in range(len(feature_words)):
            print feature_words[i].decode()
        train_feature_list,test_feature_list=TextFeatures(train_data_list,test_data_list,feature_words,flag)
        test_class_list=Text_Classifier(train_feature_list,test_feature_list,train_class_list)
        #test_accuracy_list.append(test_accuracy)
      #  if test_class_list="u'C000024' u'C000008' u'C000008'":
      #      break
        print test_class_list


#    plt.figure()
#    plt.plot(deleteNs,test_accuracy_list)
#    plt.title('Relationship of deleteNs and test_accuracy')
#    plt.xlabel('deleteNs')
#    plt.ylabel('test_accuracy')
#    plt.savefig('result.png')
#    
#    print 'finished'



    

