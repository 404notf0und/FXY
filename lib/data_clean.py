import urllib

def clean_rn(rn):
	""" clean \r\n of every string
	Parameters
	----------
	rn:str
		string of data

	Returns
	----------
	result:str
		no \r\n in string

	"""
	return rn.strip().replace('\n', '').replace('\r', '').strip()

def clean_repeat(rp):
	""" clean repeat of list
	Parameters
	----------
	rp:list
		list of data

	Returns
	----------
	result:list
		no repeat in list
	"""
	return list(set(rp))

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