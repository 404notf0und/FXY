# FXY：***Security-Scenes-Feature-Engineering-Toolkit***

![](https://img.shields.io/badge/python-3.7-red) 
![](https://img.shields.io/github/license/404notf0und/fxy) 
![](https://img.shields.io/badge/Security%20Scenes-2-green)
![](https://img.shields.io/badge/Feature%20Methods-3-blue)

特征工程框架，用于安全场景中 **数据预处理|分析|特征化|向量化** 等任务。

# 特点
- 支持多种安全场景的特征化
- 内置多种通用特征化方法
- 内置脚本扩展支持快速二次开发

# 依赖
- Python3.7
- Pip

# 安装

    git clone https://github.com/404notf0und/FXY
    pip install -r requirement.txt

# 用法

	from feature_vec.data import data_load
	from feature_vec.nlp2vec import tfidf,wordindex,word2vec
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import classification_report

	x1,y1,x2,y2=data_load('part1.csv','part3.csv')
	nlp=word2vec(one_class=True,vocabulary_size=300,out_dimension=2)
	fx1,fy1=nlp.fit_transform(x1,y1)
	fx2=nlp.transform(x2)

	lr=LogisticRegression()
	lr.fit(fx1,fy1)
	r2=lr.predict(fx2)
	print(classification_report(y2,r2))

# 用户手册
- 需求与设计
- 快速开始
- 代码结构
- 二次开发

# 贡献
1. 提交issues
2. 邮件联系[root@4o4notfound.org](root@4o4notfound.org)
3. 欢迎提交安全数据向量化的脚本，您贡献的方法将被共享并致谢

# 其他
- [问题反馈](https://github.com/404notf0und/FXY/issues/new)
- [版权声明](https://github.com/404notf0und/FXY/blob/master/LICENSE)

# 联系作者
- mail：[root@4o4notfound.org](root@4o4notfound.org)
