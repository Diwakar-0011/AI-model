from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x,y=load_iris(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Accuracy",model.score(x_test,y_test))