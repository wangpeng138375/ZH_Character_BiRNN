#! usr/bin/python
#coding=utf-8 
'''
Created on Feb 7, 2017

@author: wangpeng
'''



"""

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329 name, out_type

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""


import time,os,sys

import numpy as np
import tensorflow as tf

import charactertrain_birnn_softlayer_add.reader_placeholder_sa as reader
import utils.configuration as configuration
import utils.writer 
import utils.costom_rnn as constmrnn

staticconfig={
        "batch_size":4,
        "vocab_size":12,
        "embedding_size":6,
        "dropout":1,
        "num_layers":2,
        "name":"../modelresult/ptb_word_small_sentence",
        "log":"../log/ptb_word_small_sentence.log",
        "save_path":"../modelresult",
        "data_path":"../data/testfolder",
        "layer":"LSTM",
        "learning_rate":1,
        "max_epoch":4,
        "max_max_epoch":10,
        "init_scale":0.1,
        "max_grad_norm":5,
        "lr_decay":0.5,
        "forget_bias":0,
        "optimizer":"sgd"

    
    }
mask_voc_embedding=[[0.]*staticconfig["vocab_size"],[1.]*staticconfig["vocab_size"]]


def data_type():
    return tf.float32
# can't calculate the 3-dim because of the embedding dimision is not zeros
# def length(sequence):
#     mabs=tf.abs(sequence)
#     print mabs
#     used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
#     print used ,"=== used"
#     length = tf.reduce_sum(used, reduction_indices=1)
#     length = tf.cast(length, tf.int32)
#     return length
def length(sequence):
    mabs=tf.abs(sequence)
    print mabs
    used = tf.sign(tf.abs(sequence))
    print used ,"=== used"
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length-1#这里减去1，可以直接把brnn中的最后一个<s>或者</s>去掉

class BiRNNLM(object):
    def __init__(self, is_training,config):# configuration in train,valid , and test is different
        
        mask_voc_embedd=tf.convert_to_tensor(mask_voc_embedding,dtype=tf.float32)
        
        with tf.name_scope("InputData"):
            self.inputX = tf.placeholder(tf.int32, [config["batch_size"], None]) # [batch_size, num_steps]
            print (self.inputX.name,"-=-=-=-=------------------------------------")
            #seqlen = tf.placeholder(tf.int32, [config["batch_size"]])
            self.inputY = tf.placeholder(tf.int32, [config["batch_size"],None])
            print (self.inputY.name,"-=-=-=-=------------------------------------")

        size = config['embedding_size']
        vocab_size = config['vocab_size']
        self.seqlen=length(self.inputY)
        print (self.seqlen)
        
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
            embedding_fw = tf.get_variable(
                "embedding_fw", [vocab_size, size], dtype=data_type())
            embedding_bw = tf.get_variable(
                "embedding_bw", [vocab_size, size], dtype=data_type())
            input_emb_x = tf.nn.embedding_lookup(embedding_fw, self.inputX)
            input_emb_y = tf.nn.embedding_lookup(embedding_bw, self.inputY)
            print (input_emb_x.name,"===============================input_emb_x")
            print (input_emb_y.name,"===============================input_emb_y")
        

        
        
        if is_training and config['dropout'] < 1:
            input_emb_x = tf.nn.dropout(input_emb_x, config['dropout'])
            input_emb_y = tf.nn.dropout(input_emb_y, config['dropout'])
        
        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        print (self.seqlen,"sssssssssssssssssssssssss")
        outputs, (state_fw,state_bw) = constmrnn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_emb_x, input_emb_y,
                                                                     initial_state_fw=self.fw_initial_state, initial_state_bw=self.bw_initial_state,
                                                                     sequence_length=self.seqlen)
        generating_mask = tf.sign(tf.reduce_max(tf.abs(outputs[0]), reduction_indices=2))
        
        generated_mask_y=tf.cast(generating_mask, tf.int32)
        generating_mask=tf.reshape(generated_mask_y,[ -1])
        print (generating_mask,",,,,,,,,,,,,,,,,",outputs[0])
        generated_mask=tf.nn.embedding_lookup(mask_voc_embedd, generating_mask)
        print (outputs,"oooooooooooooooooooooooooooooo")
        outputs_added=outputs[0]+outputs[1]
        
        # for every element in output , such as outputs[0] , its shape is [batch_size,hidden_size]
        # so the output is 
        temp1=tf.concat(1, outputs)
        print (temp1,"ttttttttttttttttttttttttttttttttttttemp1")
#         temp=tf.concat(2, outputs)
#         print (temp,"ttttttttttttttttttttttttttttttttttttemp")
        output = tf.reshape(tf.concat(1, outputs_added), [-1, size])
        print (output,"pppppppppppppppppppppppppppppppppppppppppp")
        
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        print(logits,"llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
        logits*=generated_mask
        print(logits,"llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
        #mask = tf.sign(tf.reduce_max(tf.abs(output), reduction_indices=2))
        #self.inputY_sliced=tf.slice(self.inputY, [0,0], size, name),
        label=self.inputY*generated_mask_y
        print (label,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabel")
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(label, [-1])],
            [tf.ones([config["batch_size"] * tf.reduce_max(self.seqlen+1)], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / config["batch_size"]
        self._fw_final_state = state_fw
        self._bw_final_state = state_bw
        
        if not is_training:
            return
        
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config['max_grad_norm'])
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

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
#     @property
#     def seqlen(self):
#         return self.seqlen




def run_epoch(session, model, eval_op=None, verbose=False,batch_class=None):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  
  
  
  state_fw = session.run(model.fw_initial_state)
  state_bw = session.run(model.bw_initial_state)

  fetches = {
      "cost": model.cost,
      "fw_final_state": model.fw_final_state,
      "bw_final_state": model.bw_final_state,
      "inputx":model.input_x,
      "inputy":model.input_y,
      "seqlen":model.seqlen,

#         "output1":"Train/Model/BiRNN/FW/FW/transpose:0",
#         "output2":"Train/Model/ReverseSequence:0",
#         "logits":"Train/Model/mul:0",
#         "mask":"Train/Model/Reshape:0",
#         "label":"Train/Model/mul_1:0",

  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(batch_class.epoch_size):
    input_x,input_y=batch_class.next_batch()
    temp_num_step=len(input_x[0])
#     print(input_x,"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     print (input_y,"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy temp_num_step",temp_num_step)
    feed_dict = {}


    feed_dict[model.input_x]=input_x
    feed_dict[model.input_y]=input_y
#     print(input_x,"============================================",input_y)
    
    for i, (c, h) in enumerate(model.fw_initial_state):
      #print("state----------fw",i,c,state_fw[i].c)
      feed_dict[c] = state_fw[i].c
      feed_dict[h] = state_fw[i].h
    for i, (c, h) in enumerate(model.bw_initial_state):
      #print("state----------bw",i,c,state_bw[i].c)
      feed_dict[c] = state_bw[i].c
      feed_dict[h] = state_bw[i].h
    #print("============================================")

    vals = session.run(fetches, feed_dict)

#     print ("========================> inputx ",vals["inputx"].shape,vals["inputx"])
#     print ("========================> inputy ",vals["inputy"].shape,vals["inputy"])
#     print ("========================> label ",vals["label"].shape,vals["label"])
#     print ("========================> seqlen ",vals["seqlen"])
#     print ("========================> output2 ",vals["output2"].shape,vals["output2"])
#     print ("========================> logits ",vals["logits"].shape,vals["logits"])
#     print ("========================> mask ",vals["mask"].shape,vals["mask"])


    
    #############################################
    cost = vals["cost"]
    #################### control the switch to next iteration ###############
    #state_fw = vals["fw_final_state"]
    #state_bw = vals["bw_final_state"]

    costs += cost
    iters += temp_num_step
    #print (step,batch_class.epoch_size,temp_num_step,verbose,batch_class.epoch_size // 10)

    if verbose and step % (batch_class.epoch_size // 10) == 4:
      print("%.3f perplexity: %.3f speed: %.0f wps" % 
            (step * 1.0 / batch_class.epoch_size, np.exp(costs / iters),
             iters * batch_class.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)





def main(_):

    if os.path.isfile('{0}.final'.format(staticconfig['name'])):
        raise StandardError("{0}.final already exists. If you want to re-train the model, remove the model file and its checkpoints.".format(staticconfig['name']))

    fout = file(staticconfig['log'],'w')
    sys.stdout = utils.writer.writer(sys.stdout, fout)

    print('configuration:')
    for par,value in staticconfig.iteritems():
        print('{0}\t{1}'.format(par, value))

    eval_config = staticconfig.copy() # same parameters for evaluation, except for:
    eval_config['batch_size'] = 1 # batch_size
    eval_config['num_steps'] = 1 # and number of steps
    
    
    all_data, id_to_word, vocabulary_length = reader.ptb_raw_data(staticconfig)
    train_data = all_data[0]
    valid_data = all_data[1]
    test_data = all_data[2]
    train_batches=reader.Batches(train_data,staticconfig["batch_size"])
    valid_batches=reader.Batches(valid_data,staticconfig["batch_size"])
    test_batches=reader.Batches(test_data,eval_config["batch_size"])

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-staticconfig['init_scale'],
                                                staticconfig['init_scale'])

        with tf.name_scope("Train"):
            
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtrain = BiRNNLM(is_training=True,config=staticconfig)
            tf.scalar_summary("Training Loss", mtrain.cost)
            tf.scalar_summary("Learning Rate", mtrain.lr)
    
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = BiRNNLM(is_training=False,config=staticconfig)
            tf.scalar_summary("Validation Loss", mvalid.cost)
    
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = BiRNNLM(is_training=False,config=eval_config)
    
        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            #session.run(tf.global_variables_initializer())
            for i in range(staticconfig['max_max_epoch']):
                lr_decay = staticconfig['lr_decay'] ** max(i + 1 - staticconfig['max_epoch'], 0.0)
                mtrain.assign_lr(session, staticconfig['learning_rate'] * lr_decay)
                
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
                train_perplexity = run_epoch(session, mtrain, eval_op=mtrain.train_op,
                                             verbose=True,batch_class=train_batches)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid,batch_class=train_batches)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            
            test_perplexity = run_epoch(session, mtest,batch_class=test_batches)
            print("Test Perplexity: %.3f" % test_perplexity)
            
            if staticconfig["save_path"]:
              print('saving final model to {0}.final'.format(staticconfig['name']))
              sv.saver.save(session, '{0}.final'.format(staticconfig['name']))


if __name__ == "__main__":
  tf.app.run()
