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
	# print("[+] Starting Load module "+imp_module)
	try:
		lib = importlib.import_module(imp_module)
	except:
		print("[!] load feature process module error")
		exit()
	print("[+] Loaded module "+imp_module)

	# good_file=args.good_samples_filename
	# bad_file=args.bad_samples_filename
	train_all_filename=args.train
	test_all_filename=args.test

	if train_all_filename:
		cs=lib.demo()
		all_samples=pd.read_csv(data+train_all_filename,header=0,names=['payload','label'])
		x,y=cs.fxy_train(all_X_samples=all_samples['payload'],all_Y_samples=all_samples['label'])
		#model=cs.model_train(x,y)
		if test_all_filename:
			all_samples=pd.read_csv(data+test_all_filename,header=0,names=['payload','label'])
			x,y=cs.fxy_test(all_X_samples=all_samples['payload'],all_Y_samples=all_samples['label'])
			#model=cs.model_train(x,y)
	else:
		# true_X_samples=pd.read_csv(data+good_samples,names=['payload'])
		# false_X_samples=pd.read_csv(data+bad_samples,names=['payload'])
		# cs=lib.demo(true_X_samples=true_X_samples['payload'],true_Y_samples=None,false_X_samples=false_X_samples['payload'],false_Y_samples=None,all_X_samples=None,all_Y_samples=None)
		# x,y=cs.fxy() #level="word"
		# model=cs.train(x,y)
		print("[!] trainning data Not Found")

if __name__=="__main__":
	main()

