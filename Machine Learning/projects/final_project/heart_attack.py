""" 
	Developed under ICT academy IIT KANPUR.
	Developed by Pankaj Kumar(NIT UTTARAKHAND)

"""

try:
	# import all necessary module.
	import numpy as np
	import pandas as pd
	import joblib
	import time
	import warnings
	import seaborn as sns
	import matplotlib.pyplot as plt
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
	from sklearn.metrics import classification_report
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import KFold
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.naive_bayes import GaussianNB,BernoulliNB
	from sklearn.decomposition import PCA
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.impute import SimpleImputer 
	from sklearn.svm import SVC
	warnings.filterwarnings(action="ignore")
	

	try:
		# student info section.

		print("Developed by Pankaj Kumar under ICT academy IIT KANPUR.")
		print("My email is : pankajkumarmath1@gmail.com")
		print("order id is : 369632")


		# now get data from csv file
		filename = "../../datasets/heart_attack_data.csv"
		whole_data = pd.read_csv(filename)
		print("\n\nTop five row of data looks like \n\n",whole_data.head(5))
		col_names=whole_data.columns
		print("\n\nName of all the columns of data is \n\n ",col_names)
		print("\n\nShape of data is ",whole_data.shape)

		# last column name contain some space so i rename it.
		last_column = []
		# get last column name in last_column variable
		for i in col_names:
			last_column = i
			# renaming last column with "target"
		whole_data.rename(columns = {last_column:"target"},inplace = True)
		col_names = whole_data.columns
		print("\n\nAfter renaming name of all the columns of data is \n\n",col_names)
		print("\n\nDescription of data is \n",whole_data.describe(),"\n\n")


		# here description show about 5 column.
		# it means rest 9 column contain some null or string or some thing else
		# so we have to chack it first

		# check for null value in data
		# if any one is find then we have to replace it with either mean/median/mode.
		# or if frequecy of null is more then we can remove that column as well.
		for i in col_names:
			print(f"  Column is {i} and number of null value in it is {whole_data[i].isnull().sum()} ")

		# this give output zero for all column
		# means none column has none.Its means some other value are in column.
		# now check and analizing every column value and their frequency.

		for i in col_names:
			print("\nColumn is ",i)
			print("\nunique values in data is ",whole_data[i].unique())
			print("\nvalue with their frequecy \n",whole_data.groupby(i).size())
			whole_data[i].hist()
			plt.title(f"Histogram for {i} column of data.")
			plt.show()

		# show 9 column contain ? 
		# trestbps contain 1 ?
		# chol contain 23 ?
		# fbs contain 8 ?
		# restecg contain 1 ?
		# thalach contain 1 ?
		# exang contain 1 ?
		# slope contain 190 ?
		# ca contain 291 ?
		# thal contain 266 
		# so we have to remove(slope,ca and thal columns) these two columns first because it very less data.

		# now drop these three column
		del whole_data['ca']
		del whole_data['slope']
		del whole_data['thal']

		# here all 6 column contain very less no of ? symbol
		# so we can not remove whole column.
		# we have to replace it.
		# first replace with np.nan value and change dtypes of all column
		whole_data.replace('?',np.nan,inplace=True)
		whole_data=whole_data.astype('float64')
		# except chol column i replace all the ? by mode value and chol column with mean value. 

		change_list = ['trestbps','fbs','restecg','thalach','exang']
		for i in change_list:
			mode_value = whole_data[i].mode()[0]
			whole_data[i] = whole_data[i].fillna(mode_value)
			# print to clarify that data is replace completely.
			print("unique in",whole_data.groupby(i).size())

		mean_value=round(whole_data['chol'].mean())
		whole_data['chol'] = whole_data['chol'].fillna(mean_value)
		print("unique in ",whole_data.groupby('chol').size())


		# now analyse the data with help of some graph and 
		# try to reduce those column whose dependency is near to 1(one).

		pd.set_option('display.width', 1000)
		pd.set_option('display.max_column', 35)
		correlation = whole_data.corr(method = 'pearson')
		print("\nKarl pearson correleation coefficient \n",correlation)
		print('\n\n\n')

		# from correlation data we analyze that none column has dependency with anyone near 1(one).
		# so we can not remove any column.

		# now plot density graph for more information
		whole_data.plot(kind = 'density', subplots = True, layout = (3,4), sharex = False,legend=False)
		# plt.legend()
		plt.title("density plot of data " )
		plt.show()
		

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		cax = ax1.imshow(whole_data.corr() )
		ax1.grid(True)
		plt.title('Attributes of Correlation')
		# Add colorbar, make sure to specify tick locations to match desired ticklabels
		fig.colorbar(cax)
		plt.show()


		# now extract data from dataframe for model.

		Y = whole_data['target'].values
		X = whole_data.drop('target',axis=1).values

		X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=1)

		# now declare some varible for algorithum.

		list_of_lagorithum = []
		num_of_folds = 10
		results_of_algo = []
		names_of_algo = []

		# now check accuracy with some algo without standardization.
		list_of_lagorithum.append(('RandomForestClassifier ',RandomForestClassifier()))
		list_of_lagorithum.append(('QuadraticDiscriminantAnalysis ',QuadraticDiscriminantAnalysis()))
		list_of_lagorithum.append(('LogisticRegression ',LogisticRegression()))
		list_of_lagorithum.append(('DecisionTree ',DecisionTreeClassifier()))
		list_of_lagorithum.append(('LinearDiscriminant ',LinearDiscriminantAnalysis()))
		list_of_lagorithum.append(('Support Vector Machine ',SVC()))
		list_of_lagorithum.append(('GaussianNB ',GaussianNB()))
		list_of_lagorithum.append(('BernoulliNB ',BernoulliNB()))
		list_of_lagorithum.append(('KNeighborsClassifier ',KNeighborsClassifier()))


		print("\n\n\nAccuracies of algorithm without  standardization \n\n")
		for name, model in list_of_lagorithum:
			kfold = KFold(n_splits=num_of_folds, random_state=13)
			startTime = time.time()
			cv_results = cross_val_score(model, X,Y, cv=kfold, scoring='accuracy')
			endTime = time.time()
			results_of_algo.append(cv_results)
			names_of_algo.append(name)
			print( f"{name}: { cv_results.mean()*100}% with deviation of {cv_results.std()*100} and (run time: {endTime-startTime} sec",end="\n\n")

		fig = plt.figure()
		fig.suptitle('Performance Comparison')
		ax = fig.add_subplot(111)
		plt.boxplot(results_of_algo, showfliers=False)
		ax.set_xticklabels(names_of_algo)
		plt.show()


		# now analysis of result for further improvement


		# acuuracy of above mention algorithum.



		# RandomForestClassifier : 65.96551724137932% with deviation 
									# of 22.788987635655356 and (run time: 1.1766512393951416 sec
		# QuadraticDiscriminantAnalysis : 78.48275862068967% with deviation 
									# of 11.991739791469703 and (run time: 0.009546756744384766 sec
		# LogisticRegression : 79.47126436781609% with deviation 
									# of 13.586887588214847 and (run time: 0.1671152114868164 sec
		# DecisionTree : 57.79310344827585% with deviation 
									# of 20.28348437538922 and (run time: 0.013154029846191406 sec
		# LinearDiscriminant : 80.47126436781609% with deviation 
									# of 14.844279118209315 and (run time: 0.013259649276733398 sec
		# Support Vector Machine : 61.758620689655174% with deviation 
									# of 42.58015065441808 and (run time: 0.025760889053344727 sec
		# GaussianNB : 81.88505747126436% with deviation 
									# of 12.065812395991724 and (run time: 0.011449575424194336 sec
		# BernoulliNB : 78.42528735632185% with deviation 
									# of 14.309218438798036 and (run time: 0.012790679931640625 sec
		# KNeighborsClassifier : 57.89655172413792% with deviation 
									# of 20.625446343312912 and (run time: 0.02408885955810547 sec


		# form this we can conclude that GaussianNB and LDA are performing well.
		# from boxplot we can analyse that the median of LDA is more than GaussianNB
		
		# now we can conclude that LDA is more better than all(without standardization)



		# now apply standardization and then analyse whether the accuracy is increase or not

		pipelines_with_standardization = []
		names_with_standardization = []
		results_with_standardization = []




		pipelines_with_standardization.append(('ScaledRandomForestClassifier ',Pipeline([('Scaler',\
			StandardScaler()),('RandomForestClassifier ',RandomForestClassifier())])))

		pipelines_with_standardization.append(('ScaledQuadratic ',Pipeline([('Scaler',\
			StandardScaler()),('QuadraticDiscriminantAnalysis ',QuadraticDiscriminantAnalysis())])))

		pipelines_with_standardization.append(('ScaledLogistic ',Pipeline([('Scaler',\
			StandardScaler()),('LogisticRegression ',LogisticRegression())])))

		pipelines_with_standardization.append(('ScaledDecisionTree ',Pipeline([('Scaler',\
			StandardScaler()),('DecisionTreeClassifier ',DecisionTreeClassifier())])))

		pipelines_with_standardization.append(('ScaledLDA ',Pipeline([('Scaler',\
			StandardScaler()),('LinearDiscriminantAnalysis ',LinearDiscriminantAnalysis())])))

		pipelines_with_standardization.append(('ScaledSVM ',Pipeline([('Scaler',\
			StandardScaler()),('Support Vector Machine ',SVC())])))

		pipelines_with_standardization.append(('ScaledGaussianNB ',Pipeline([('Scaler',\
			StandardScaler()),('GaussianNB ',GaussianNB())])))

		pipelines_with_standardization.append(('ScaledBernoulliNB ',Pipeline([('Scaler',\
			StandardScaler()),('BernoulliNB ',BernoulliNB())])))

		pipelines_with_standardization.append(('ScaledKNeighbors ',Pipeline([('Scaler',\
			StandardScaler()),('KNeighbors ',KNeighborsClassifier())])))

		print("\n\n\nAccuracies of algorithm after scaled dataset\n")


		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			kfold = KFold(n_splits=num_of_folds,random_state=123)
			for name,model in pipelines_with_standardization:
				start=time.time()
				cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
				end = time.time()
				results_with_standardization.append(cv_results)
				names_with_standardization.append(name)
				print( f"{name} :  { cv_results.mean()*100}% with deviation of {cv_results.std()*100} and (run time: {endTime-startTime} sec",end="\n\n")


		# ScaledRandomForestClassifier  :  80.1578947368421% with deviation 
									# of 12.329839039822152 and (run time: 0.02477550506591797 sec
		# ScaledQuadratic  :  79.10526315789474% with deviation 
									# of 11.280360694415844 and (run time: 0.02477550506591797 sec
		# ScaledLogistic  :  82.18421052631581% with deviation 
									# of 9.562485855365853 and (run time: 0.02477550506591797 sec
		# ScaledDecisionTree  :  75.92105263157895% with deviation 
									# of 13.329538356514226 and (run time: 0.02477550506591797 sec
		# ScaledLDA  :  82.18421052631578% with deviation 
									# of 8.151143216388238 and (run time: 0.02477550506591797 sec
		# ScaledSVM  :  80.07894736842105% with deviation 
									# of 10.620662712575598 and (run time: 0.02477550506591797 sec
		# ScaledGaussianNB  :  81.6842105263158% with deviation 
									# of 9.325002877766854 and (run time: 0.02477550506591797 sec
		# ScaledBernoulliNB  :  83.21052631578947% with deviation 
									# of 9.490336242434186 and (run time: 0.02477550506591797 sec
		# ScaledKNeighbors  :  81.10526315789473% with deviation 
									# of 7.87172083598419 and (run time: 0.02477550506591797 sec




		fig = plt.figure()
		fig.suptitle('Performance Comparison after Scaled Data')
		ax = fig.add_subplot(111)
		plt.boxplot(results_with_standardization)
		ax.set_xticklabels(names_with_standardization)
		plt.show()

		# after aplling standardization technique we see that the accuracy of all algorithum is 
		# increased and after then we conclude that till now ScaledBernoulli are doing best among all
		# so we select this for fureture process.


		# now prepare model and fit teach model for our prediction.
		scaler = StandardScaler().fit(X_train)

		X_train_scaled = scaler.transform(X_train)
		model = BernoulliNB()
		start=time.time()
		model.fit(X_train_scaled,Y_train)
		end=time.time()

		print( "\n\nBernoulliNB Training Completed. It's Run Time: %f" % (end-start))

		X_test_scaled = scaler.transform(X_test)
		predictions = model.predict(X_test_scaled)
		print("Predictions done successfully by BernoulliNB Algorithms")
		print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))
		print("\n")
		print("confusion_matrix = \n")
		cf_matrix = confusion_matrix(Y_test, predictions)
		print(cf_matrix)



		sns.heatmap(cf_matrix,annot=True,fmt='g')
		plt.show()

		# now save model for furture use of prediction

		filename =  "finalized_heart_attack_model.sav"
		joblib.dump(model, filename)
		print( "Best Performing Model dumped successfully into a file by Joblib")

		
		# student info section.

		print("Developed by Pankaj Kumar under ICT academy IIT KANPUR.")
		print("My email is : pankajkumarmath1@gmail.com")
		print("order id is : 369632")


	except:
		# error message if file not exist.
		print("Used file is not in current position or some other problem occur.")


except:
	# error message when any module are not installed in system.
	print("Some module are not installed.\nPlease install it first.")
