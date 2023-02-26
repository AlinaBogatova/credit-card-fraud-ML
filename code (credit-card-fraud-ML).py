import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# функция определения ближайшего соседа
def Neighbor (train_x, test_x, train_y, test_y, n):
    clsf = KNeighborsClassifier(n_neighbors= n)
    clsf.fit(train_x, train_y)
    y_pred = clsf.predict(test_x)
    return y_pred

#функция создания модели логистической регрессии
def log_reg (train_x, test_x, train_y):
    clf = LogisticRegression(max_iter = 2000)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred
#функция создания дерева решений
def DesTree (train_x, test_x, train_y):
    clf = DecisionTreeClassifier(criterion = 'gini', min_samples_split= 5,
                                  min_samples_leaf= 5)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred

#функция создания модели случайного леса
def Random_forest (train_x, test_x, train_y):
    clf = RandomForestClassifier(criterion='gini', min_samples_split = 5, 
                                 min_samples_leaf=5)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred 

#функция создания нейронной сети
def MLP (train_x, test_x, train_y):
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic')
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred
# функция верификации - расчет метрик точности
def verif(y_pred, test_y):
    print(accuracy_score(test_y, y_pred))
    print(recall_score(test_y, y_pred))
    print(precision_score(test_y, y_pred))
    print(f1_score(test_y, y_pred))
df = pd.read_csv('creditcard.csv', sep = ',')
df.drop(['Time'], axis = 1,  inplace = True)
print(df.duplicated().value_counts())
df = df.drop_duplicates()
print(df.duplicated().value_counts())
print(df['Class'].unique())
print(df['Class'].value_counts())
num_0 = df[df['Class'] == 0]
num_1 = df[df['Class'] == 1]
df = pd.concat([num_0.sample(len(num_1)), num_1])
df = df.reset_index(drop = True)
print(df.head())
print(df.tail())
print(df.info())
# разделение на обучающую и тестовую выборки
train_x, test_x, train_y, test_y = train_test_split(df[df.columns[:-1]],
                                                    df[df.columns[-1]], random_state = 1)

print('======================Nearest_neighbor=============')
y = Neighbor (train_x, test_x, train_y, test_y, 4)
verif(y, test_y)
print('======================Logistic regression=============')
y2 = log_reg (train_x, test_x, train_y)
verif(y2, test_y)

print('======================Desision tree=============')
y3 = DesTree (train_x, test_x, train_y)
verif(y3, test_y)

print('======================neural_network=============')
y4 = MLP (train_x, test_x, train_y)
verif(y4, test_y)

print('==========================Random forest======================')
y5 = Random_forest(train_x, test_x, train_y)
verif(y5, test_y)


