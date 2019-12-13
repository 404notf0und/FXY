# import os
# import pandas as pd
# from lib.data_clean import clean_rn
# def loadFileTxt(name):
#     """ load file from file path
#     Parameters
#     ----------
#     name:str
#         the name of file

#     Returns
#     ----------
#     result:list
#         file data

#     """
#     directory = "../data/"
#     filepath = os.path.join(directory, name)
#     with open(filepath,'r',encoding="utf8") as f:
#         data = f.readlines()

#     data = list(set(data))
#     result = []
#     for d in data:
#         d = str(d) 
#         d=clean_rn(d)  #clean \r\n
#         result.append(d)
#     return result

# #print(loadFileTxt("url.txt"))

# def loadFileCsv(name):
#     """ read file from csv using pandas
#     Parameters
#     ----------
#     name:str
#         csv filename

#     Returns
#     ----------
#     df:DataFrame
#         csv content
#     """
#     directory = "../data/"
#     filepath = os.path.join(directory, name)
#     df=pd.read_csv(filepath,names=['payload'])
#     return df

# #print(loadFileCsv("phish_url.csv"))