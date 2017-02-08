# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace(" \n", "").split()
    
def _read_words_tolist(filename):
    sent_word_list=[]
    with tf.gfile.GFile(filename, "r") as f:
        sentlist=f.read().decode("utf-8").split("\n")
        for sent in sentlist:
            sent_word_list.append(sent.split())
        return sent_word_list
        


def _build_vocab(filename):
    data = _read_words(filename)
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1,len(words)+1)))
    id_to_word = dict(zip(range(1,len(words)+1), words))
    word_to_id["@"]=0
    id_to_word[0]="@"
    if not '<UNK>' in word_to_id and not '<unk>' in word_to_id:
        word_to_id['<unk>'] = len(word_to_id)
        id_to_word[len(id_to_word)] = '<unk>'
    fdw2d=open("../data/vocab_w2d.txt","w")
    fdd2w=open("../data/vocab_d2w.txt","w")
    for w in word_to_id.items():
        fdw2d.write('{}\t{}\n'.format(w[0].encode("utf8"), w[1])) 
    for w in id_to_word.items():
        fdd2w.write('{}\t{}\n'.format(w[0], w[1].encode("utf8"))) 
    fdw2d.close()
    fdd2w.close()

    return word_to_id,id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def _file_to_sent_word_ids(filename, word_to_id):
    datalist = _read_words_tolist(filename)
    #return [word_to_id[word] if word in word_to_id else word_to_id["<unk>"] for sent in datalist  for word in sent ]
    wordidlist=[]
    for sent in datalist:
        wordidlist.append([word_to_id[word] if word in word_to_id else word_to_id["<unk>"] for word in sent ])
    return wordidlist
    


def ptb_raw_data(config=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(config['data_path'], "train.txt")
  valid_path = os.path.join(config['data_path'], "valid.txt")
  test_path = os.path.join(config['data_path'], "test.txt")

  word_to_id,id_to_word = _build_vocab(train_path)
  train_data = _file_to_sent_word_ids(train_path, word_to_id)
  valid_data = _file_to_sent_word_ids(valid_path, word_to_id)
  test_data = _file_to_sent_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return (train_data, valid_data, test_data),id_to_word, vocabulary


class Batches:
    def __init__(self, raw_data,batch_size):
        self.cursor=0
        self.raw_data=raw_data
        self.raw_data_length=len(raw_data)
        self.batch_size=batch_size
        self.batch_length=len(raw_data)//batch_size
        
    def next_batch(self):
        
        one_batch=self.raw_data[self.cursor*self.batch_size:(self.cursor+1)*self.batch_size]
        xy_padded_batch=self._padding(one_batch)
        self.cursor+=1
        if self.cursor>=self.batch_length:
            self.cursor=0
        return xy_padded_batch
    def _padding(self,data):
        maxlen=-1
        datalen=len(data)
        for sentence in data:
            slen=len(sentence)
            if maxlen<slen:
                maxlen=slen
        for sentence in data:
            num_pads = maxlen - len(sentence)
            for pos in xrange(num_pads):
                sentence.append(0)
        ndata=np.asarray(data,dtype=np.int32)
        ndata_x=ndata[0:datalen,0:maxlen-1]
        ndata_y=ndata[0:datalen,1:maxlen]
        return ndata_x,ndata_y
    @property
    def epoch_size(self):
        return self.batch_length


if __name__ == '__main__':
    
    
    config={
        "batch_size":20,
        "vocab_size":100,
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
        "max_max_epoch":3
    
    }
    word_to_id,id_to_word = _build_vocab("../data/testfolder/train.txt")
#     print (word_to_id)
#     print (_read_words_tolist("../data/testfolder/test.txt"))
#     print (_file_to_sent_word_ids("../data/testfolder/test.txt",word_to_id))
#     train_data = _file_to_sent_word_ids("../data/testfolder/train.txt", word_to_id)
#     print (train_data)
#     a,b=batch_producer(train_data,2)
#     sv = tf.train.Supervisor()
#     with sv.managed_session() as session:
#         print (session.run([a,b]))
    all_data, id_to_word, vocabulary_length = ptb_raw_data(config)
    print (all_data[0])
    print (all_data[1])
    print (all_data[2])
    print (all_data[2][0:2])
    cla=Batches(all_data[2],2)
    print ("=============================")
    xx,yy=cla.next_batch()
    print(cla.next_batch()[0])
    print(cla.next_batch())
    print(cla.next_batch())
    print(cla.next_batch())
    print(cla.next_batch())
    print(cla.next_batch())
    minputX = tf.placeholder(tf.int32, [2, None])
    ia=minputX*2
    with tf.Session() as sess:
        print (sess.run(ia,feed_dict={minputX:xx}))

    
#     sv = tf.train.Supervisor()
#     with sv.managed_session() as sess:
#         print (sess.run(batched_data))
