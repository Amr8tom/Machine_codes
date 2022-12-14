import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from  sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

#the dataset is available on kaggle too
train = pd.read_csv("C:\\Users\\amr alaa\\Downloads\\house-votes-84.data.csv",names=["category",2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

#check for missing values
#print(train.isnull().sum())
train=train.replace(to_replace="?",value=None)
train.dropna(inplace=True)
#print(train.head())
print(train)
# I selected few of the columns from the dataset for this tutorial
train[2]=train[2].replace(to_replace="y",value=1)
train[2]=train[2].replace(to_replace="n",value=0)
train[3]=train[3].replace(to_replace="y",value=1)
train[3]=train[3].replace(to_replace="n",value=0)
train[4]=train[4].replace(to_replace="y",value=1)
train[4]=train[4].replace(to_replace="n",value=0)
train[5]=train[5].replace(to_replace="y",value=1)
train[5]=train[5].replace(to_replace="n",value=0)
train[6]=train[6].replace(to_replace="y",value=1)
train[6]=train[6].replace(to_replace="n",value=0)
train[7]=train[7].replace(to_replace="y",value=1)
train[7]=train[7].replace(to_replace="n",value=0)
train[8]=train[8].replace(to_replace="y",value=1)
train[8]=train[8].replace(to_replace="n",value=0)
train[9]=train[9].replace(to_replace="y",value=1)
train[9]=train[9].replace(to_replace="n",value=0)
train[10]=train[10].replace(to_replace="y",value=1)
train[10]=train[10].replace(to_replace="n",value=0)
train[11]=train[11].replace(to_replace="y",value=1)
train[11]=train[11].replace(to_replace="n",value=0)
train[12]=train[12].replace(to_replace="y",value=1)
train[12]=train[12].replace(to_replace="n",value=0)
train[13]=train[13].replace(to_replace="y",value=1)
train[13]=train[13].replace(to_replace="n",value=0)
train[14]=train[14].replace(to_replace="y",value=1)
train[14]=train[14].replace(to_replace="n",value=0)
train[15]=train[15].replace(to_replace="y",value=1)
train[15]=train[15].replace(to_replace="n",value=0)
train[16]=train[16].replace(to_replace="y",value=1)
train[16]=train[16].replace(to_replace="n",value=0)
train[17]=train[17].replace(to_replace="y",value=1)
train[17]=train[17].replace(to_replace="n",value=0)


x=train.drop(columns=["category"])
y=train["category"]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=50)
clf=DecisionTreeClassifier(max_depth=2)
clf.fit(X_train,y_train)

dot_data = tree.export_graphviz(clf, out_file=None,
                               feature_names=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                               class_names=['Yes','No'],filled=True,
                                rounded=True,
                              special_characters=True)
print(clf.score(X_test,y_test))
graph = graphviz.Source(dot_data)
graph.render()
graph
