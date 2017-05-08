import cv2
import numpy as np
import os
import random
import Image
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc import imsave
from sklearn.linear_model import LogisticRegression
from sklearn import svm
label_vol = {}
x_train = []
y_train = []
x_test = []
y_test = []

global_train = 15

def calcSiftFeature(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(200) # max number of SIFT points is 200
	kp, des = sift.detectAndCompute(gray, None)
	return des

def calcFeatVec(features, centers):
	featVec = np.zeros((1, 50))
	for i in range(0, features.shape[0]):
		fi = features[i]
		diffMat = np.tile(fi, (50, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1	
	return featVec

def fix_size(img , width , height) :
	out = img.resize((width , height) , Image.ANTIALIAS)
	return out
	
def Labelexp(label_vol , num , label) :
	if num not in label_vol:
		label_vol[num] = label
	return label_vol


def DataInitialize():
	cat = 0
	featureSet = np.float32([]).reshape(0,128)
	train_map = []
	train_list = {}
	test_list = {}
	for filename in os.listdir("101_ObjectCategories") :
		cur_train_list=[]
		cur_test_list=[]	
		
		featureSet = np.float32([]).reshape(0,128)
		current_path = "101_ObjectCategories/" + filename
		print "splitting category :" + filename
		Num_train = global_train
		index = [i for i in range(len(os.listdir(current_path)))]
		random.shuffle(index)
		seed_train = [index[i] for i in range(Num_train)]
		train_map.append(seed_train)
		cnt = 0
		for pic in os.listdir(current_path) :			
			if cnt not in seed_train:
				cur_test_list.append(current_path + "/" + pic)
			else :
				final_path = current_path + "/" + pic
				cur_train_list.append(current_path + "/" + pic)
				img = cv2.imread(final_path)
				des = calcSiftFeature(img)
				if des != None :
					featureSet = np.append(featureSet, des, axis=0)
					file_name = "Temp/features/" + filename + ".npy"
					np.save(file_name, featureSet)
			cnt += 1
		#print cur_test_list
		train_list[cat] = cur_train_list
		test_list[cat] = cur_test_list
		#print test_list
		cat += 1
	return train_map , train_list , test_list
			
def learnVocabulary():
	wordCnt = 50
	for name in os.listdir("101_ObjectCategories") :
		print "creating vocabulary for : " + name
		filename = "Temp/features/" + name + ".npy"
		features = np.load(filename)
		
		# use k-means to cluster a bag of features
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		compactness, labels, centers = cv2.kmeans(features, wordCnt, criteria, 20, flags)
		
		# save vocabulary(a tuple of (labels, centers)) to file
		filename = "Temp/vocabulary/" + name + ".npy"
		np.save(filename, (labels, centers))

def trainClassifier():
	train_map = np.load("train_map.npy")
	trainData = np.float32([]).reshape(0, 50)
	response = np.float32([])
	cnt = 0
	dictIdx = 0
	for name in os.listdir("101_ObjectCategories") :
		count = 0 
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		current_path = "101_ObjectCategories/" + name
		seed_train = train_map[cnt]
		print "Init training data of " + name + "..."
		for pic in os.listdir(current_path) :
			if count in seed_train :
				final_path =  current_path + "/" + pic
				img = cv2.imread(final_path)
				#img = fix_size(img , 360 ,240)
				features = calcSiftFeature(img)
				if features != None :
					featVec = calcFeatVec(features, centers)
					trainData = np.append(trainData, featVec, axis=0)
					response = np.append(response, np.float32([dictIdx]))
			count +=1
		
		#res = np.repeat(np.float32([dictIdx]), test_num)
		
		dictIdx += 1
		cnt += 1

	
	trainData = np.float32(trainData)
	response = response.reshape(-1, 1)
	np.save("trainData.npy",trainData)
	np.save("response.npy",response)
	

if __name__ == "__main__":	
	train_map , train_list , test_list = DataInitialize()
	#np.save("train_map.npy" , train_map)
	#np.save("train_list.npy",train_list)
	#np.save("test_list.npy",test_list)
	
	learnVocabulary()
	
	trainClassifier()
	#train_list = np.load("train_list.npy")
	#test_list = np.load("test_list.npy")
	#print type(test_list)
	#print test_list
	trainData = np.load("trainData.npy")
	response = np.load("response.npy")
	response = np.ravel(response)
	clf = svm.SVC(kernel='linear', C=1)
	clf.fit(trainData,response)
	
	#train_map = np.load("train_map.npy")
	total = 0; correct = 0; dictIdx = 0 ; cnt = 0
	for name in os.listdir("101_ObjectCategories") :
		crt = 0
		labels, centers = np.load("Temp/vocabulary/" + name + ".npy")
		#current_path = "101_ObjectCategories/" + name
		#test_num = len(os.listdir(current_path)) - 20
		#seed_train = train_map[cnt]
		count = 0
		cc = 0
		print "Classify on TestSet " + name + ":"
		if len(test_list[cnt]) >= 30 :
			for pic in test_list[cnt][:30] :
				#final_path =  current_path + "/" + pic
				img = cv2.imread(pic)
				features = calcSiftFeature(img)
				if features != None :
					featVec = calcFeatVec(features, centers)
					case = np.float32(featVec)
					if (dictIdx == clf.predict(case)):
						crt += 1
					cc += 1
		else :
			for pic in test_list[cnt] :
				#final_path =  current_path + "/" + pic
				img = cv2.imread(pic)
				features = calcSiftFeature(img)
				if features != None :
					featVec = calcFeatVec(features, centers)
					case = np.float32(featVec)
					if (dictIdx == clf.predict(case)):
						crt += 1
					cc += 1	
						
		cnt += 1
		print "Accuracy: " + str(crt) + " / " + str(cc) + "\n"
		correct += crt
		total += cc
		dictIdx += 1
		
	print "Total accuracy: " + str(correct) + " / " + str(total)
	
	
	
	
