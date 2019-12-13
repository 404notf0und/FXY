import urllib
import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def urlparse(url):
	"""converting url encoded data to simple string
	Parameters
	----------
	url:str
		string of url

	Returns
	----------
	unquote_url:str
		unquote of url

	"""
	unquote_url=urllib.parse.unquote(url)
	return unquote_url

def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=unquote(unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
