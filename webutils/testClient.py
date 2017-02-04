# encoding=utf-8
'''
Created on 2017-01-07
@author: user
'''
import urllib2
import urllib
import json
def test_getprob_get(sentence):
    url = 'http://172.17.197.61:8080/getprob?'
    data = {"data":sentence}     
    new_url = url + urllib.urlencode(data)
    print new_url

    response = urllib2.urlopen(new_url)
    resultstr = response.read()
    resultstr=resultstr.replace("'","\"")
    print resultstr,type(resultstr)
    resultDict=json.loads(resultstr)
    if resultDict["status"]=="true":
        return json.loads(resultDict["prob"])
    return None

if __name__ == '__main__':
    # test_init_get()
    # import os
    # print os.getcwd()
    # test_init_get()
    
    sentence="<s> 我 喜 欢 你 你 知 道 吗 </s>"
    r=test_getprob_get(sentence)
    print r
    for i in r:
        print i
    
    #test_init_post()