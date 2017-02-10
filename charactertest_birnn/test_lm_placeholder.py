#! usr/bin/python
#coding=utf-8 
'''
Created on Feb 8, 2017

@author: wangpeng
'''

import time,os,sys

import numpy as np
import tensorflow as tf

import charactertrain_birnn.reader_placeholder as reader
import utils.configuration as configuration
import utils.writer 

testconfig={
        "batch_size":1,
        "vocab_size":12,
        "embedding_size":10,
        "dropout":1,
        "num_layers":2,
        "name":"../modelresult/ptb_word_small_sentence",
        "log":"../log/ptb_word_small_sentence.log",
        "save_path":"../modelresult",
        "data_path":"../data/testfolder",
        "layer":"LSTM",
        "learning_rate":1,
        "max_epoch":4,
        "max_max_epoch":3,
        "init_scale":0.1,
        "max_grad_norm":5,
        "lr_decay":0.5,
        "forget_bias":0,
        "optimizer":"sgd",
        "test":1,
        "lm":"../modelresult/ptb_word_small_sentence.final",
        "result":"../log/out"

    
    }

def data_type():
    return tf.float32

def length(sequence):
    mabs=tf.abs(sequence)
    print mabs
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    print used ,"=== used"
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

class BiRNNLM(object):
    def __init__(self, is_training,config):# configuration in train,valid , and test is different
        
        with tf.name_scope("InputData"):
            self.inputX = tf.placeholder(tf.int32, [config["batch_size"],None]) # [batch_size, num_steps]
            print (self.inputX,"-=-=-=-=------------------------------------")
            self.inputY = tf.placeholder(tf.int32, [config["batch_size"],None])
            print (self.inputY,"-=-=-=-=------------------------------------")
            

        size = config['embedding_size']
        vocab_size = config['vocab_size']
        
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config['dropout'] < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config['dropout'])
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config['num_layers'], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config['num_layers'], state_is_tuple=True)
        
        self._fw_initial_state = fw_cell.zero_state(config["batch_size"], data_type())
        self._bw_initial_state = bw_cell.zero_state(config["batch_size"], data_type())
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self.inputX)
            print (inputs.name,"===============================22222222")
        seqlen=length(inputs)
    
        
        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        print( inputs)
        print (seqlen,"sssssssssssssssssssssssss")
        outputs, (state_fw,state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, 
                                                                     initial_state_fw=self.fw_initial_state, initial_state_bw=self.bw_initial_state,
                                                                     sequence_length=seqlen)
        print (outputs,"oooooooooooooooooooooooooooooo")
        
        # for every element in output , such as outputs[0] , its shape is [batch_size,hidden_size]
        # so the output is 
        temp1=tf.concat(1, outputs)
        print (temp1,"ttttttttttttttttttttttttttttttttttttemp1")
        temp=tf.concat(2, outputs)
        print (temp,"ttttttttttttttttttttttttttttttttttttemp")
        output = tf.reshape(tf.concat(1, outputs), [-1, size*2])
        print (output,"pppppppppppppppppppppppppppppppppppppppppp")
        
        softmax_w = tf.get_variable(
            "softmax_w", [size*2, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        # normalize scores -> probabilities
        softmax_output = tf.nn.softmax(logits)
        
        print (tf.shape(softmax_output),softmax_output.name,"softmax_outputsoftmax_outputsoftmax_outputsoftmax_outputsoftmax_outputsoftmax_output")

#         reshaped = tf.reshape(softmax_output, [vocab_size*4]) 
#         print (tf.shape(reshaped),reshaped.name,"9999999999999999999999999999")
        ####################################gather_nd#################################
        #    result=tf.gather_nd(temp, index)
        #    temp:    [[ 1 11 21 31 41]
        #             [51 61 71 81 91]]
        #
        #    index:    [[0 2]
        #                [1 3]]
        #
        #    result:    [21 81]
        ##############################################################################
        words_count=tf.range(tf.shape(softmax_output)[0])
        gather_index=tf.pack([words_count,tf.squeeze(self.inputY)], 1)
        # get probability of target word
        # the gather:
        # temp = [ 1 11 21 31 41 51 61 71 81 91] 
        # tg.gather(temp,[1,5,9])
        # ===>[11 51 91]
        prob_tensor = tf.gather_nd(softmax_output, gather_index) 
        print (tf.shape(prob_tensor),prob_tensor.name)
        #print(prob_tensor)
        self._target_prob = prob_tensor 


        self._fw_final_state = state_fw
        self._bw_final_state = state_bw


        # do not update weights if you are not training

        return

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    

    @property
    def input_x(self):
        return self.inputX
    @property
    def input_y(self):
        return self.inputY

    @property
    def fw_initial_state(self):
        return self._fw_initial_state
    @property
    def bw_initial_state(self):
        return self._bw_initial_state
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def fw_final_state(self):
        return self._fw_final_state
    @property
    def bw_final_state(self):
        return self._bw_final_state

    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    @property
    def target_prob(self):
        return self._target_prob


def load_vocab(config):
    '''Returns a word-to-id and id-to-word (or character-to-id and id-to-character) mapping 
    for all words (or characters) in filename.'''
    
    if "test" in config:
        print("************************test phase ***********************")
        fdw2d=open("../data/vocab_w2d.txt","r")
        w2d={}
        d2w={}
        for line in fdw2d:
            word, wid = line.strip().split()
            w2d[word]=int(wid)
            d2w[int(wid)]=word
        fdw2d.close()
        return (w2d,d2w)
def sentence_to_word_ids(sentence, word_to_id):
    '''Returns list of all words in the file, either one long list or a list of lists per sentence.'''

    data_ids = []
    #data_ids.append([word_to_id[word] for word in sentence if word in word_to_id])# no <unk> for now
    data_ids=[word_to_id[word] if word_to_id.has_key(word) else word_to_id["<unk>"] for word in sentence ]
    print data_ids,sentence,"sentence_to_word_ids--------------------"
    return [data_ids]

def getCandidateSentence(id2words,softmatrix):
    pass

def getProbability(session,mtest, words,vocabulary,config=None):
    if not words:
        return None
    word_to_id=vocabulary[0]

    words = words.split(" ")

    words_id= sentence_to_word_ids(words, word_to_id)
    print words_id,"====================words_id"

    probResult=getProbability_Sentence(session,mtest,words_id)
    return probResult[0]

def getProbability_Sentence(session, model,raw_data1):
    """Runs the model on the given data."""
    
    print (raw_data1,"============================raw_data11111",type(raw_data1[0]))
#     all_vars = tf.trainable_variables()
#     for v in all_vars:
#         print (v.name)
    
    inputlen=len(raw_data1[0])-1
    print (inputlen,"=======================inputlen")
    # it is useless , just for histroy reason
    num_words = 1
    # state = initial state of the model
    
    state_fw = session.run(model.fw_initial_state)
    state_bw = session.run(model.bw_initial_state)
    resultList=[]
    # fetches = what the graph will return
    fetches = {
        #"cost": model.cost,
        
        "input_sample": model.input_x,
        "target_sample": model.input_y,
        "target_prob": model.target_prob,
        "softout":"Test/Model/Softmax:0",
        #"softreshape":"Test/Model/Shape_1:0"


#         "rawsoft": "Test/Model/Softmax:0",
#         "shapedsoft":"Test/Model/Reshape_1:0",
#         "gatherpro":"Test/Model/Gather:0"

    }

    


    #print ("step",step)
    #print(raw_data1)
    #print(raw_data1[step:step+2])
    
    #rawsoft,shapedsoft,gatherpro=session.run(["Test/Model/Softmax:0","Test/Model/Reshape_1:0","Test/Model/Gather:0"])
    np_raw_data=np.asarray(raw_data1)

    feed_dict={}
    print np_raw_data[0:1,0:inputlen]
    print np_raw_data[0:1,1:inputlen+1],np_raw_data[0:1,1:inputlen+1].shape
    feed_dict[model.input_x]=np_raw_data[0:1,0:inputlen]
    feed_dict[model.input_y]=np_raw_data[0:1,1:inputlen+1]

    for i, (c, h) in enumerate(model.fw_initial_state):

        feed_dict[c] = state_fw[i].c
        feed_dict[h] = state_fw[i].h
    for i, (c, h) in enumerate(model.bw_initial_state):
        feed_dict[c] = state_bw[i].c
        feed_dict[h] = state_bw[i].h

    # feed the data ('feed_dict') to the graph and return everything that is in 'fetches'
    vals = session.run(fetches, feed_dict)

    print "================================================="
    print vals["softout"],"90909090909090000000000000000000000000000"
    #print vals["softreshape"]
    print "++++++++++++========================+++++++++++++"
    # debugging: print every sample (input + target) that is fed to the model
#         if PRINT_SAMPLES:
#             print_samples(vals['input_sample'], vals['target_sample'], id_to_word)
        

    target_prob = vals["target_prob"]
    #print (target_prob,"==============================================ttrtradfasdfasdfasfakpropro")
    resultList.append(target_prob)
    return resultList,vals["softout"]








def main(_):

    vocab=load_vocab(testconfig)
    print (vocab,"====================////////////////////////")

    with tf.Graph().as_default():  
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=None):
                mtest = BiRNNLM(is_training=False,config=testconfig)

        # sv = training helper that checkpoints models and computes summaries
        
        #sv = tf.train.Supervisor(logdir=config['save_path'])
        saver = tf.train.Saver()
    
        # managed_session launches the checkpoint and summary services
        with tf.Session() as session:
    
            print (testconfig['lm'])
            # restore variables from disk
            saver.restore(session, testconfig['lm'])
            
            print('Start rescoring...')
            #run_epoch(session, mtest, id_to_word, out)
            #rs=getProbability(session,mtest,"<s> a b c </s>",vocab)
#             print (rs,"=======================================result")
#             rs=getProbability(session,mtest,"<s> a b c zz </s>",vocab)
#             print (rs,"=======================================result")
#             rs=getProbability(session,mtest,"<s> b c a </s>",vocab)
#             print (rs,"=======================================result")
#             rs=getProbability(session,mtest,"<s> a b d </s>",vocab)
#             print (rs,"=======================================result")
            rs=getProbability(session,mtest,"<s> a b c g e f g </s>",vocab)
            print (rs,"=======================================result")
            rs=getProbability(session,mtest,"<s> a b c d e f g </s>",vocab)
            print (rs,"=======================================result")

            
if __name__ == "__main__":
    tf.app.run()

