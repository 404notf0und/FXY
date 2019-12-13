#2017/05/27
#Ref:https://github.com/surajr/URL-Classification
#Need to repaired
from urllib.parse import urlparse
import tldextract
import pandas as pd
import ipaddress as ip
from keras.utils import to_categorical
import numpy as np

class demo:
	def __init__(self,true_X_samples=None,true_Y_samples=None,false_X_samples=None,false_Y_samples=None,all_X_samples=None,all_Y_samples=None):
		# self.function=url_countdots
		#self.allfunction=[self.url_countdots,self.url_countdelim]
	# 	self.countdots=self.url_countdots
	# 	self.countdelim=self.url_countdelim
	# 	self.isip=self.url_isip
	# 	self.isPresentHyphen=self.url_isPresentHyphen
	# 	self.isPresentAt=self.url_isPresentAt
	# 	self.isPresentDSlash=self.url_isPresentDSlash
	# 	self.countSubDir=self.url_countSubDir
	# 	self.all=[self.url_countdelim,self.url_countdots,self.url_isip,self.url_isPresentHyphen,self.url_isPresentAt,self.url_isPresentDSlash,self.url_countSubDir]
	# # Method to count number of dots
		print("[+] Init module url_phishing_lexical")
		self.all_X_samples=all_X_samples
		self.all_Y_samples=all_Y_samples
		self.featureSet = pd.DataFrame(columns=('no of dots','presence of hyphen','len of url','presence of at',\
						'presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP'))
	def fxy(self):
		"""get features x and y
		Parameters
		----------
		self.all_X_samples:DataFrame
		self.all_Y_samples:DataFrame

		Returns
		----------
		X:DataFrame
			columns:10
		Y:numpy.ndarray
			columns:classes num
		"""
		print("[+] Start feature engineering")
		for i in range(len(self.all_X_samples)):
			features=[]
			features = self.getFeatures(self.all_X_samples.loc[i])
			self.featureSet.loc[i] = features
		X=self.featureSet
		Y=self.all_Y_samples.values
		Y=to_categorical(Y)
		#Y=np.array(Y)
		print("[+] Done!Successfully got Feature X Y")
		print("[+] Fxy shape:",X.shape,Y.shape)
		return X,Y
	
	def getFeatures(self,url):
		result = []
		url = str(url)
		path = urlparse(url)
		#print(path)
		ext = tldextract.extract(url)
		#print(ext)
		result.append(self.url_countdots(ext.subdomain))
		result.append(self.url_isPresentHyphen(path.netloc))
		result.append(len(url))
		result.append(self.url_isPresentAt(path.netloc))
		result.append(self.url_isPresentDSlash(path.path))
		result.append(self.url_countSubDir(path.path))
		result.append(self.url_countSubDomain(ext.subdomain))
		result.append(len(path.netloc))
		result.append(len(path.query))
		result.append(self.url_isip(ext.domain))
		#result.append(str(label))
		#print(result)
		#result.append(1 if ext.suffix in Suspicious_TLD else 0)
		return result

	def url_countdots(self,url): 
		return url.count('.')

	def url_countSubDomain(self,subdomain):
	    if not subdomain:
	        return 0
	    else:
	        return len(subdomain.split('.'))
	# Method to count number of delimeters
	def url_countdelim(self,url):
		count = 0
		delim=[';','_','?','=','&']
		for each in url:
			if each in delim:
				count = count + 1
		
		return count

	# Is IP addr present as th hostname, let's validate
	import ipaddress as ip #works only in python 3
	def url_isip(self,uri):
		try:
			if ip.ip_address(uri):
				return 1
		except:
			return 0

	#method to check the presence of hyphens
	def url_isPresentHyphen(self,url):
		return url.count('-')

	#method to check the presence of @
	def url_isPresentAt(self,url):
		return url.count('@')

	def url_isPresentDSlash(self,url):
		return url.count('//')

	def url_countSubDir(self,url):
		return url.count('/')