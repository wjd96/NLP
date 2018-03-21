import tensorflow as tf
import numpy as np
import jieba
import gensim
import pandas as pd
import sys
import re

w2v_model_save="d:/1/save_nn/model"
vector_size=100
data_path='d:/1/save_nn/data.csv';
sentence_size=30
layer_num=1
batch_size=20000
train_iter=10
train_model_path="d:/1/save_nn/train.ckpt"
pattern='[A-Z]{1,}[0-9]{4}'
stop_word=[' ',',']
data_size=52693
class W2v_nn(object):
    def __init__(self,w2v_model_save,vector_size,sentence_size,train_model_path,data_path,batch_size):
        self.w2v_model_save=w2v_model_save
        self.vector_size=vector_size
        self.sentence_size=sentence_size
        self.train_model_path=train_model_path
        self.train_iter=train_iter
        self.data_path=data_path
        self.batch_size=batch_size
        print("load_data...")
        self.load_data_process_w2v()
        self.sample_f_index=0
        self.sample_b_index=0
    def cut_word(self,data):
        sentences=[]
        for i in data:
            mark=re.search(pattern,i)
            lt=re.split(pattern,i,maxsplit=1)
            sentence=[]
            if len(lt)>=2:
                sentence1=list(jieba.cut(lt[0]))
                sentence2=list(jieba.cut(lt[1]))
                sentence=np.concatenate([sentence1,[mark.group()],sentence2])
            else:
                sentence=list(jieba.cut(i))
            sentence=[j for j in sentence if j not in stop_word]
#             print(sentence)
            sentences.append(sentence)
        return sentences
    def load_data_process_w2v(self):
        data=pd.read_csv(self.data_path,encoding="gbk",header=None)
        self.total_sample=data_size
        self.sentences1=self.cut_word(data.ix[:,0].values)
        self.sentences2=self.cut_word(data.ix[:,1].values)
        self.y=data.ix[:,2].values
#         return sentences1,sentences2,data.ix[:,2].values;           
    def get_train_batch(self):
        if self.sample_b_index==self.total_sample:
            self.sample_f_index=self.sample_b_index=0
        if self.sample_b_index+self.batch_size<self.total_sample:
            self.sample_f_index=self.sample_b_index
            self.sample_b_index=self.sample_f_index+self.batch_size
        else:
            self.sample_f_index=self.sample_b_index
            self.sample_b_index=self.sample_f_index+(self.total_sample-self.sample_f_index)
        vectors1=[]
        vectors2=[]
        for i in self.sentences1[self.sample_f_index:self.sample_b_index]:
            i=list(set(i))
            sum=np.zeros(self.vector_size)
            for j in i:
                if j in self.w2v_model:
                    sum=sum+self.w2v_model[j]
            sum=sum/len(i)
            vectors1.append(sum)
        for i in self.sentences2[self.sample_f_index:self.sample_b_index]:
            i=list(set(i))
            sum=np.zeros(self.vector_size)
            for j in i:
                if j in self.w2v_model:
                    sum=sum+self.w2v_model[j]
            sum=sum/len(i)
            vectors2.append(sum)
        y=self.y[self.sample_f_index:self.sample_b_index]
        return vectors1,vectors2,y
#W2V
    def trainW2v(self):
        sentences=np.concatenate([self.sentences1,self.sentences2])
        model=gensim.models.Word2Vec(sentences,size=self.vector_size,window=4,min_count=1)
        model.save(self.w2v_model_save)
        
    def get_w2v(self):
        self.w2v_model=gensim.models.Word2Vec.load(w2v_model_save)
        
    def get_distence(self):
        v1,v2,_=self.get_train_batch()
        dist=[]
        for i in range(0,len(v1)):
            dist.append(np.sqrt(np.sum(np.square(v1[i]-v2[i]))))
        return dist
    
    def nn(self):
        self.y_=tf.placeholder(tf.float32,[None],'y')
        self.x=tf.placeholder(tf.float32, [None,vector_size], 'x')
        weight=tf.Variable(tf.random_normal([vector_size,1]));
        bias=tf.Variable(tf.random_normal([1]));
        s_c=tf.nn.sigmoid(tf.matmul(self.x,weight)+bias,name='compara')
        s_c=tf.reshape(s_c,[-1])
        loss=tf.nn.l2_loss(tf.subtract(self.y_, s_c,),'loss')
        apply_gradient=tf.train.AdamOptimizer(0.01).minimize(loss)
        return loss,apply_gradient
    
    def train(self):
        loss,gradient=self.nn()
        sess=tf.Session()
        initial=tf.global_variables_initializer()
        sess.run(initial)
        saver=tf.train.Saver()
        for i in range(0,100):
            v1,v2,y=self.get_train_batch()
            print(len(v1))
            print(self.sample_f_index,self.sample_b_index)
            print(self.sample_b_index-self.sample_f_index)
            sub=[]
            for i in range(0,self.sample_b_index-self.sample_f_index):
                sub.append(v1[i]-v2[i])
            for i in range(0,train_iter):
                l,g=sess.run([loss,gradient],feed_dict={self.x:sub,self.y_:y})
                print("batch:",self.sample_b_index/self.batch_size,"  loss:",l)
        saver.save(sess, train_model_path)
#         pass
vn=W2v_nn(w2v_model_save,vector_size,sentence_size,train_model_path,data_path,batch_size)
# print("trian word2voc...")
# vn.trainW2v()
print("initial word2voc model...")
vn.get_w2v()
#  
# print("train_data...")
# vn.train()

# v1,v2,y=vn.get_train_batch()
# print("compute distince...")
# dist=vn.get_distence()
# pf=pd.DataFrame();
# pf['0']=y;
# pf['1']=dist
# print("write...")
# pf.to_csv("d:/1/save_nn/res.csv",header=False,index=False)


# y=np.reshape(y,[-1,1])
# dist=np.reshape(dist,[-1,1])
# print(np.concatenate([y,dist],axis=1))

sess = tf.Session()
saver = tf.train.import_meta_graph(train_model_path+".meta")
saver.restore(sess,train_model_path)
graph = tf.get_default_graph() 
s_t=graph.get_tensor_by_name("compara:0")
inputs1=graph.get_tensor_by_name("x:0")
# inputs2=graph.get_tensor_by_name("x2:0")
#     y=graph.get_tensor_by_name("y:0")
batch_1,batch_2,y=vn.get_train_batch()
#     with tf.Graph().as_default() as g:
r=sess.run(s_t,feed_dict={inputs1:batch_1})
print(r)
pf=pd.DataFrame();
pf['0']=y
pf['1']=r
print("write...")
pf.to_csv("d:/1/save_nn/res1.csv",header=False,index=False)
