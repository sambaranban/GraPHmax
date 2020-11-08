GraPHmax:
Datasets:
There is a folder graphdatasets that contains all the required datasets.
How to run:
Use the following command to run GraPHmax with default values of the parameters on MUTAG dataset. 
python main.py 
Requirements:
Install python in your system. Requirements file has been given in the model folder.  To install the same libraries run
	pip install -r requirements.txt		
These are python libraries that are required to run the code.
1)	Tensorflow (version: 1.14.0) with python 3.6.9.
2)	Networkx (version: 2.3)
3)	matplotlib
4)	pandas
5)	keras
6)	scipy
7)	pickle
8)	collections
Hyperparameters Settings:
There are some hyperparameters one can change during execution. 
1)	plrt : Pooling ratio
2)	plly : Number of Pooling layers
3)	emb : Embedding dimenion
4)	lrp : Learning rate for periphery representation
5)	lrh : Learning rate for hierarchical representation
6)	lrl : Learning rate for classification task
7)	dropout : Dropout rate
8)	negpr : Number of negative samples for periphery discriminator
9)	neghr: Number of negative samples hierarchical discriminator
10)	dataset: Name of the dataset to perform classification on.

One can specify these hyperparameters during running the code.
	python main.py --dataset MUTAG
