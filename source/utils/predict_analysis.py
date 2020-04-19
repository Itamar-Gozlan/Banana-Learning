import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys

tgt_size = (336, 252) # for single plant
# tgt_size = (252, 1008) # triplets original size - GPU exhausted
# tgt_size = (151, 504) # triplets new size
path_validation = '/home/itamargoz/data/sorted/seg/validation'


def index_to_category(index, labels):
	switch = {0: "A", 1: "B", 2: "C", 3: "D"}
	return switch.get(index)

def print_dictionary(dict):
	for key in sorted (dict) : 
		print(str(key), ' : ', str(dict[key]))
	return sum(dict.values())

def increase(dict, key):
	if dict.get(key):
		dict[key] = dict[key] + 1
	else:
		dict[key] = 1

# def count_pred_by_date(path, categories, model_name):
# 	model_path = "/home/itamargoz/trunk/Banana-Learning/saved_models/"+model_name+".h5"
# 	model = load_model(model_path)
# 	true_pre = {}
# 	false_pre = {}
# 	for dir in os.listdir(path):
# 		if categories.get(dir) == False : continue
# 		for sdir in os.listdir(path+"/"+dir):
# 			img_path = path + "/"+  dir+ "/" + sdir
# 			img = image.load_img(img_path, target_size = tgt_size)
# 			img = image.img_to_array(img)
# 			img = np.expand_dims(img, axis = 0)
# 			y_res = index_to_category(np.argmax(model.predict(img)))
# 			date, y_true = sdir.split("_")[-8], sdir.split("_")[-4]
# 			print(sdir + " : " + "y_true = " + str(y_true) + " y_res = " + str(y_res) + " : " + 
# 				str(y_res==y_true) + " Prediction : Date " + date)
# 			increase(true_pre, date) if y_res == y_true else increase(false_pre, date)
			
# 	return true_pre, false_pre

def count_pred_by_date(path, model_name, labels):
	model_path = "/home/itamargoz/trunk/Banana-Learning/saved_models/old_saved/"+model_name+".h5"
	model = load_model(model_path)
	true_pre = {}
	false_pre = {}
	for dir in os.listdir(path):
		for sdir in os.listdir(path+"/"+dir):
			img_path = path + "/"+  dir+ "/" + sdir
			img = image.load_img(img_path, target_size = tgt_size)
			img = image.img_to_array(img)
			img = np.expand_dims(img, axis = 0)
			y_res = labels[np.argmax(model.predict(img))]
			date, y_true = sdir.split("_")[-8], sdir.split("_")[-4]
			print(sdir + " : " + "y_true = " + str(y_true) + " y_res = " + str(y_res) + " : " + 
				str(y_res==y_true) + " Prediction : Date " + date)
			increase(true_pre, date) if y_res == y_true else increase(false_pre, date)
	return true_pre, false_pre

def predict_analysis_one_model(model_name, labels):
	print("================== PREDICTION SUMMARY FOR " + model_name + " ==================")
	true_pre, false_pre = count_pred_by_date(path_validation,model_name, labels)
	print("========= TRUE PREDICTION BY DATE =========")
	n_true = print_dictionary(true_pre)
	print("========= FALSE PREDICTION BY DATE =========")
	n_false = print_dictionary(false_pre)
	print("========= TOTAL PREDICTION SUMMARY " + model_name + "=========")
	print("total : " + str(n_true + n_false))
	print("total true: " + str(n_true))
	print("total false: " + str(n_false))
	print("accuracy = " +  str(n_true/(n_true + n_false)))



# python ./predict_analysis.py [model_name] [A,B,..]
# if len(sys.argv) < 2:
# 	print("Error! model name was not defined")
# 	print("format: /predict_analysis.py [model_name] [A,B,..]")
# 	exit()

# model_name = sys.argv[1]
# labels = sys.argv[2].split(",")
# predict_analysis_one_model(model_name + "_NO_AUG", labels)
# predict_analysis_one_model(model_name + "_YES_AUG", labels)

# "/home/itamargoz/trunk/Banana-Learning/saved_models/CIFAR-10-SEG_FULL-WA.h5"
# predict_analysis_one_model(model_name, labels)
labels = ['A','B','C','D']
predict_analysis_one_model("CIFAR-10-AVSALL-NA_1", labels)
predict_analysis_one_model("CIFAR-10-AVSALL-WA_1", labels)


