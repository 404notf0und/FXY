import argparse
import pandas as pd
import importlib

data='.\\data\\'

def main():
	parser = argparse.ArgumentParser(description="Fxy V0.1")

	parser.add_argument("-s", "--scenes", help="security scenes")
	parser.add_argument("-f", "--features", help="features engine")
	#parser.add_argument("-m", "--train_or_test", help="trainning or testing")
	# parser.add_argument("-g", "--good_samples_filename", help="good_samples")
	# parser.add_argument("-b", "--bad_samples_filename", help="bad_samples")
	parser.add_argument("-ta", "--train", help="all_samples_filename_trainning")
	parser.add_argument("-tb", "--test", help="all_samples_filename_testing")

	args = parser.parse_args()

	scene=args.scenes
	feature=args.features
	imp_module='feature_lib.'+scene+'_'+feature
	#imp_module='feature_lib.'+feature
	try:
		lib = importlib.import_module(imp_module)
	except:
		print("[!] Load module %s error" %imp_module)
		exit()
	print("[+] Load module "+imp_module)

	# good_file=args.good_samples_filename
	# bad_file=args.bad_samples_filename
	train_all_filename=args.train
	test_all_filename=args.test

	if train_all_filename:
		cs=lib.demo()
		pre_x,pre_y=cs.pre_processing(train_all_filename)
		x,y=cs.fxy_train(pre_x,pre_y)
		#model=cs.model_train(x,y)
	elif test_all_filename:
		pre_x,pre_y=cs.pre_processing(test_all_filename)
		x,y=cs.fxy_test(pre_x,pre_y)
		#model=cs.model_test(x,y)
	else:
		print("[!] trainning and testing data Not Found")

if __name__=="__main__":
	main()

