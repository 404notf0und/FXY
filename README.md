# FXY：***Security-Scenes-Feature-Engineering-Toolkit***

[![](https://img.shields.io/badge/python-3.7-red)](https://www.python.org/) [![](https://img.shields.io/github/license/404notf0und/fxy)](https://github.com/404notf0und/FXY/blob/master/LICENSE) [![](https://img.shields.io/badge/Support%20Security%20Scenes-7%2B-green)](https://github.com/404notf0und/FXY/tree/master/data) [![](https://img.shields.io/badge/Feature%20Methods-3%2B-yellow)](https://github.com/404notf0und/FXY/blob/master/feature_vec/nlp2vec.py) [![](https://img.shields.io/github/stars/404notf0und/FXY)](https://github.com/404notf0und/FXY/stargazers)

[中文文档](https://github.com/404notf0und/FXY/blob/master/CN-README.md)

FXY is an open-sourced feature engineering framework for data preprocessing, feature engineering, model training and testing in multiple security scenes.

# Features
- Supports the feature engineering of multiple security scenarios
- Built-in multiple universal feature engineering methods
- Built-in multiple deep learning models
- Pipline jobs

# Requirements
- Python3.7
- Pip

# Installation
	git clone https://github.com/404notf0und/FXY
    pip install -r requirement.txt

# Usage
	from feature_vec.nlp2vec import tfidf,wordindex,word2vec
	nlp=word2vec(one_class=False,tunning=True,punctuation='concise')
	fx1,fy1=nlp.fit_transform(x1,y1)
	fx2=nlp.transform(x2)

# Documentation
- [Demand and Design](https://github.com/404notf0und/FXY/blob/master/docs/%E9%9C%80%E6%B1%82%E5%92%8C%E8%AE%BE%E8%AE%A1.md)
- [Quick Start](https://github.com/404notf0und/FXY/blob/master/FXY.py)
- [Development](https://github.com/404notf0und/FXY/blob/master/%E8%80%83%E8%99%91%E5%92%8C%E8%A7%A3%E5%86%B3%E7%9A%84%E9%97%AE%E9%A2%98.md)

# How to Contribute
1. Check for open issues or open a fresh issue to start a discussion around a feature idea or a bug
2. Email me [root@4o4notfound.org](root@4o4notfound.org)
3. Welcome to submit Feature-Engineering scripts,The method you contributed will be made public.

# Other
- [Issues](https://github.com/404notf0und/FXY/issues/new)
- [License](https://github.com/404notf0und/FXY/blob/master/LICENSE)

# Contact
- mail：root@4o4notfound.org
