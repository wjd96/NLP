import tensorflow as tf
import numpy as np
import jieba
import gensim
import pandas as pd
import sys
import re

w2v_model_save="d:/1/save_rnn/model"
vector_size=100
data_path='d:/1/save_rnn/data.csv';
sentence_size=30
layer_num=1
batch_size=2000
train_iter=10
train_model_path="d:/1/save_rnn/trainmodel.ckpt"
pattern='[A-Z]{1,}[0-9]{4}'
stop_word=[' ',',']
data_size=52693
class W2v_rnn(object):
    def __init__(self,w2v_model_save,vector_size,sentence_size,train_model_path,data_path,batch_size):
        self.w2v_model_save=w2v_model_save
        self.vector_size=vector_size
        self.sentence_size=sentence_size
        self.train_model_path=train_model_path
        self.train_iter=train_iter
        self.data_path=data_path
        self.batch_size=batch_size
        self.sample_f_index=0
        self.sample_b_index=0
        print("load_data...")
        self.load_data_process_w2v()
        
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
        batch_x1=[]
        batch_x2=[]
        t_c=np.zeros([vector_size])
        for i in self.sentences1[self.sample_f_index:self.sample_b_index]:
            sentence=[]
            for j in range(0,self.sentence_size):
                if j>=len(i):
                    sentence.append(t_c)
                    continue
                if(i[j] in self.w2v_model):
                    sentence.append(np.array(self.w2v_model[i[j]]))
#             length=len(sentence)
#             while(length<self.sentence_size):
#                 sentence.append(t_c)
#                 length=length+1
            batch_x1.append(sentence)
        for i in self.sentences2[self.sample_f_index:self.sample_b_index]:
            sentence=[]
            for j in range(0,self.sentence_size):
                if j>=len(i):
                    sentence.append(t_c)
                    continue
                if(i[j] in self.w2v_model):
                    sentence.append(np.array(self.w2v_model[i[j]]))
#             length=len(sentence)
#             while(length<self.sentence_size):
#                 sentence.append(t_c)
#                 length=length+1
            batch_x2.append(sentence)
        y=self.y[self.sample_f_index:self.sample_b_index]
        return batch_x1,batch_x2,y
#W2V
    def trainW2v(self):
        sentences=np.concatenate([self.sentences1,self.sentences2])
        model=gensim.models.Word2Vec(sentences,size=self.vector_size,window=4,min_count=1)
        model.save(self.w2v_model_save)
        
    def get_w2v(self):
        self.w2v_model=gensim.models.Word2Vec.load(w2v_model_save)
        
    def rnn_network(self):
        self.inputs1=tf.placeholder(tf.float32,[None,sentence_size,vector_size],name='x1')
        self.inputs2=tf.placeholder(tf.float32,[None,sentence_size,vector_size],name='x2')
        self.y_=tf.placeholder(tf.float32,[None],name='y')    
        input1=tf.transpose(self.inputs1, [1,0,2])
        input1=tf.reshape(input1,[-1,vector_size]);
        input1=tf.split(input1,sentence_size);
         
        input2=tf.transpose(self.inputs2, [1,0,2])
        input2=tf.reshape(input2,[-1,vector_size]);
        input2=tf.split(input2,sentence_size);
        
        with tf.variable_scope("net1"):
            cell_fw_1=tf.contrib.rnn.BasicLSTMCell(vector_size, forget_bias=1.0, state_is_tuple=True)
            cell_bw_1=tf.contrib.rnn.BasicLSTMCell(vector_size, forget_bias=1.0, state_is_tuple=True)
            output1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw_1, cell_bw_1,inputs=input1,  dtype=tf.float32)
        with tf.variable_scope("net2"):
            cell_fw_2=tf.contrib.rnn.BasicLSTMCell(vector_size, forget_bias=1.0, state_is_tuple=True)
            cell_bw_2=tf.contrib.rnn.BasicLSTMCell(vector_size, forget_bias=1.0, state_is_tuple=True)
            output2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw_2, cell_bw_2,inputs=input2,  dtype=tf.float32)
#         print(len(output1))
#         print(output1[-1].shape)
        output_sum1=tf.reduce_sum(output1,0)
        output_sum2=tf.reduce_sum(output2,0)
    #     print(output_sum1)
        output=tf.concat([output_sum1,output_sum2], 1)
#         print(output.shape)
        weight=tf.Variable(tf.random_normal([vector_size*4,1]))
        bias=tf.Variable(tf.random_normal([1]))
        s_c=tf.nn.sigmoid(tf.matmul(output,weight)+bias,name='computer')
        s_c=tf.reshape(s_c,[-1])
        sub=tf.subtract(self.y_,s_c)
        total_loss=tf.nn.l2_loss(sub)
#         apply_gradient=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(total_loss)
        apply_gradient=tf.train.AdamOptimizer(learning_rate=0.00001).apply_gradients(zip(tf.gradients(total_loss,tf.trainable_variables()),tf.trainable_variables()))
        return apply_gradient,total_loss,s_c
    
    def train(self):
        apply_gradient,total_loss,s_c=self.rnn_network()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver= tf.train.Saver()
        for i in range(0,100):
            x_batch1,x_batch2,y_batch=self.get_train_batch()
            print(self.sample_f_index,self.sample_b_index)
#             print(self.total_sample)
#             print(len(x_batch1))
    #         print(np.array(x_batch1).shape)
            for i in range(0,train_iter):
                _,loss,r=sess.run([apply_gradient,total_loss,s_c],feed_dict={self.inputs1:x_batch1,self.inputs2:x_batch2,self.y_:y_batch})
                print("batch:",self.sample_b_index/self.batch_size,"  loss:",loss)
        saver.save(sess,train_model_path)    
        
        
vn=W2v_rnn(w2v_model_save,vector_size,sentence_size,train_model_path,data_path,batch_size)

# print("trian word2voc...")
# vn.trainW2v()
print("initial word2voc model...")
vn.get_w2v()

print("train_data...")
vn.train()
        
