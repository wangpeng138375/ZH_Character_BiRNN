# encoding=utf-8
'''
Created on 2017-01-07
@author: user
'''

import sys,os
# curr_dir = os.path.dirname(__file__)
# root = os.path.join(curr_dir,"..")
# sys.path.append(root)
import web
import numpy as np
import tensorflow as tf
import utils.configuration as configuration
import utils.writer as writer
import charactertest_birnn_softlayer_add.test_lm_placeholder_sa as gp
import json
import traceback  


urls = ("/getprob", "Recommander_GetProb","/getallprob", "Recommander_GetAllProb",)

#eval_config = configuration.get_config(sys.argv[2])


#vocab=gp.load_vocab(eval_config)
vocab=gp.load_vocab(gp.testconfig)

with tf.Graph().as_default():  
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=None):
                mtest = gp.BiRNNLM(is_training=False,config=gp.testconfig)
        saver = tf.train.Saver()
        session=tf.Session()
saver.restore(session, gp.testconfig['lm'])

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class Recommander_GetProb:
    def __init__(self):
        pass

    def GET(self):
        try:
            sentence=web.input()
            #print "sentence :",sentence,sentence.get("data")
            rs=gp.getProbability(session,mtest,sentence.get("data").encode("utf8"),vocab)
            #rs=rs.tolist()
            #print rs,type(rs),type(rs[0]),"dddddddddddddddd",json.dumps(rs,cls=MyEncoder)
            return {"status": "true","prob":json.dumps(rs[0],cls=MyEncoder)}
        except:
            traceback.print_exc()
            return {"status": "false"}

    def POST(self):
        print "POST Method"
        
        
class Recommander_GetAllProb:
    def __init__(self):
        pass

    def GET(self):
        try:
            sentence=web.input()
            #print "sentence :",sentence,sentence.get("data")
            rs=gp.getProbability(session,mtest,sentence.get("data").encode("utf8"),vocab)
            #rs=rs.tolist()
            #print rs,type(rs),type(rs[0]),"dddddddddddddddd",json.dumps(rs,cls=MyEncoder)
            return {"status": "true","prob":json.dumps(rs[1],cls=MyEncoder)}
        except:
            traceback.print_exc()
            return {"status": "false"}

    def POST(self):
        print "POST Method"

if __name__ == '__main__':

    app = web.application(urls, globals())
    app.run()