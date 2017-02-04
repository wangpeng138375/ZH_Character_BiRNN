#! usr/bin/python
#coding=utf-8 
'''
Created on Jan 4, 2017

@author: wangpeng
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os, sys

import numpy as np
import tensorflow as tf
import utils.configuration as configuration
import utils.writer as writer
import copy


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", None,"Configuration file")
FLAGS = flags.FLAGS

# turn this switch on if you want to see the mini-batches that are being processed
PRINT_SAMPLES = True 

# turn this switch on for debugging
DEBUG = True

def debug(string):
    if DEBUG:
        sys.stderr.write('DEBUG: {0}'.format(string))


class LM(object):
    """Word- or character-level LM."""

    def __init__(self, config):
        with tf.name_scope("SentData"):
            dataids=tf.placeholder(tf.int32, None, name="sentence")
            print(dataids.name,"=========================================dataids")
        data_len=tf.shape(dataids)[0]
        dataids=tf.reshape(dataids,[1,-1])

        self._input_sample = tf.slice(dataids,[0,0],[1,1])
        self._target_sample = tf.slice(dataids,[0,1],[1,1])
        print (data_len.name,self._input_sample.name,self._target_sample.name,"=================1000")

        batch_size = 1
        num_steps = 1
        size = config['word_size']
        vocab_size = config['vocab_size']

        if config['layer'] == 'LSTM':
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=config['forget_bias'], state_is_tuple=True)
        else:
            raise ValueError("Only LSTM layers implemented so far. Set layer = LSTM in config file.")


        # multiple hidden layers
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config['num_layers'], state_is_tuple=True)
        

        # for a network with multiple LSTM layers, 
        # initial state = tuple (size = number of layers) of LSTMStateTuples, each containing a zero Tensor for c and h (each batch_size x size) 
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        print (self._initial_state,'1111111111111111')

        # embedding lookup table
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            inputs = tf.nn.embedding_lookup(embedding, self._input_sample)
        print (inputs.name,"===============================22222222")

        # failed becase of the python int of num_steps
        inputs = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, num_steps, inputs)]


        # feed inputs to network: outputs = predictions, state = new hidden state
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])

        # output layer weights
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

        # get scores
        logits = tf.matmul(output, softmax_w) + softmax_b

        # normalize scores -> probabilities
        softmax_output = tf.nn.softmax(logits)
        
        print (tf.shape(softmax_output),softmax_output.name)

        reshaped = tf.reshape(softmax_output, [vocab_size]) 
        print (tf.shape(reshaped),reshaped.name,"9999999999999999999999999999")

        # get probability of target word
        # the gather:
        # temp = [ 1 11 21 31 41 51 61 71 81 91] 
        # tg.gather(temp,[1,5,9])
        # ===>[11 51 91]
        prob_tensor = tf.gather(reshaped, self._target_sample[0,:]) 
        print (tf.shape(prob_tensor),prob_tensor.name)
        #print(prob_tensor)
        self._target_prob = tf.reduce_sum(prob_tensor) 


        self._final_state = state


        # do not update weights if you are not training

        return


        


    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

#     @property
#     def cost(self):
#         return self._cost

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def input_sample(self):
        return self._input_sample

    @property
    def target_sample(self):
        return self._target_sample

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def target_prob(self):
          return self._target_prob

def print_samples(input_sample, target_sample, id_to_word):
    ''' For debugging purposes: if PRINT_SAMPLES = True, print each sample that is given to the model.'''
    print('input_sample:',)
    for row in input_sample:
        for col in row:
            print('{0} '.format(id_to_word[col]), end="")
        print('')
    print('target_sample:',)
    for row in target_sample:
        for col in row:
            print('{0} '.format(id_to_word[col]), end="")
        print('')



def getProbability_Sentence(session, model,raw_data1):
    """Runs the model on the given data."""
    
    print (raw_data1,"============================raw_data11111",type(raw_data1[0]))
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print (v.name)
    epoch_size=len(raw_data1)-1
    # it is useless , just for histroy reason
    num_words = 1
    # state = initial state of the model
    
    state = session.run(model.initial_state)
    mystate=copy.deepcopy(state) 
    resultList=[]
    # fetches = what the graph will return
    fetches = {
        #"cost": model.cost,
        "final_state": model.final_state, # c and h of previous time step (for each hidden layer)
        "input_sample": model.input_sample,
        "target_sample": model.target_sample,
        "target_prob": model.target_prob,
        "datalen":"Test/Model/strided_slice:0",
        "input":"Test/Model/Slice:0",
        "target":"Test/Model/Slice_1:0"
#         "rawsoft": "Test/Model/Softmax:0",
#         "shapedsoft":"Test/Model/Reshape_1:0",
#         "gatherpro":"Test/Model/Gather:0"

    }

    

    for step in range(epoch_size):
        print ("step",step)
        print(raw_data1)
        print(raw_data1[step:step+2])
        
        #rawsoft,shapedsoft,gatherpro=session.run(["Test/Model/Softmax:0","Test/Model/Reshape_1:0","Test/Model/Gather:0"])

        feed_dict = {"Test/Model/SentData/sentence:0":raw_data1[step:step+2]}

        for i, (c, h) in enumerate(model.initial_state):
            if num_words==0:
                feed_dict[c] = mystate[i].c
                feed_dict[h] = mystate[i].h
            else:
                feed_dict[c] = state[i].c
                #print("state----------c",c,state[i].c)
                feed_dict[h] = state[i].h
                #print("state----------h",h,state[i].h)

        # feed the data ('feed_dict') to the graph and return everything that is in 'fetches'
        vals = session.run(fetches, feed_dict)

        # debugging: print every sample (input + target) that is fed to the model
#         if PRINT_SAMPLES:
#             print_samples(vals['input_sample'], vals['target_sample'], id_to_word)
            
        state = vals["final_state"]
        target_prob = vals["target_prob"]
        print (target_prob,"==============================================ttrtradfasdfasdfasfakpropro")
        resultList.append(target_prob)


    return resultList
    #return total_log_probs
def sentence_to_word_ids(sentence, word_to_id):
    '''Returns list of all words in the file, either one long list or a list of lists per sentence.'''

    data_ids = []
    #data_ids.append([word_to_id[word] for word in sentence if word in word_to_id])# no <unk> for now
    data_ids.append([word_to_id[word] if word_to_id.has_key(word) else word_to_id["<unk>"] for word in sentence ])
    return data_ids
def load_vocab(config):
    '''Returns a word-to-id and id-to-word (or character-to-id and id-to-character) mapping 
    for all words (or characters) in filename.'''
    
    if "nbest" in config:
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

def getProbability(session,mtest, words,vocabulary,config=None):
    if not words:
        return None
    word_to_id=vocabulary[0]
    id_to_word=vocabulary[1]
    words = words.split(" ")
    words_id= sentence_to_word_ids(words, word_to_id)
    print (words_id[0],"=============words_is ")
#     with tf.name_scope("GetProb"):
#         inputData = inputLM(config=config, data=words_id, name="InputData")
#         with tf.variable_scope("Model", reuse=None):
#             mtest = LM(config=config, input_=inputData)
    probResult=getProbability_Sentence(session,mtest,words_id[0])
    return probResult

        

def main(_):
    if FLAGS.config == None:
        raise ValueError("Please specify a configuration file.")
    else:
        config = configuration.get_config(FLAGS.config)

    fout = file(config['log'],'w')
    sys.stdout = writer.writer(sys.stdout, fout)

    print('configuration:')
    for par,value in config.iteritems():
        print('{0}\t{1}'.format(par, value))

    eval_config = config.copy() # same parameters for evaluation, except for:

    vocab=load_vocab(eval_config)

    with tf.Graph().as_default():  
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=None):
                mtest = LM(config=eval_config)

        # sv = training helper that checkpoints models and computes summaries
        
        #sv = tf.train.Supervisor(logdir=config['save_path'])
        saver = tf.train.Saver()
        session=tf.Session()

    
        # managed_session launches the checkpoint and summary services

    
    print (config['lm'])
    # restore variables from disk
    saver.restore(session, config['lm'])
    
    print('Start rescoring...')
    #run_epoch(session, mtest, id_to_word, out)
    rs=getProbability(session,mtest,"<s> a b c </s>",vocab)
    print (rs,"=======================================result")
    rs=getProbability(session,mtest,"<s> a b c zz </s>",vocab)
    print (rs,"=======================================result")

            
if __name__ == "__main__":
    tf.app.run()