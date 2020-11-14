
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import svm, metrics, model_selection
from sklearn.preprocessing import LabelEncoder
import datetime
import os

from keras.models import Sequential
from keras.layers import Dense

def setting(model_name):
    # 모델 저장 폴더 설정
    MODEL_DIR = './model/' + model_name
    #print(MODEL_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    return 1


v_data = '20201114_v1'  # 사용할 데이터 셋
filename = "./dataset/" + v_data + ".csv"
model_name = v_data

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)

df_pre = pd.read_csv(filename, header=0)  # CSV파일을 불러오는 함수를 이용

# print(df_pre.info())
features = len(df_pre.columns) - 1
# 데이터 내부의 기호를 숫자로 변환하기--- (※2)

df_pre = df_pre.sample(frac=1)

# 재학생 데이터만 활용
#df_pre = df_pre[df_pre['entYn'] == 0]

# 재학생 데이터만 활용
#df_pre = df_pre[df_pre['weekseq'] == 15]


# 학과코드를 onhot encoding 함##########################
le = LabelEncoder()
le.fit(df_pre['dept'])
df_pre['dept'] = le.transform(df_pre['dept'])

# le.fit(df_pre['gender'])
# df_pre['gender'] = le.transform(df_pre['gender'])

# le.fit(df_pre['area'])
# df_pre['area'] = le.transform(df_pre['area'])

# 재학생 데이터만 활용
#df_pre = df_pre[df_pre['entYn'] == 0]

# 재학생 데이터만 활용
#df_pre = df_pre[df_pre['weekseq'] == 15]
#df_pre = df_pre[df_pre['yearterm'] == '20191']

# data and label 분리
dataset = df_pre.values
X = dataset[:, 0:features]
Y = dataset[:, features]


# 테스트, 검증데이터 분할 7:3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

svm_model = svm.SVC()  # 학습시키기
svm_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_prediction = svm_model.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_prediction)
Y_prediction = svm_model.predict(X_test)
now = datetime.datetime.now()

print('SVM', now)
print('============================')
print("파일명 ", filename)
print("요소갯수 ", features)
print("================================== ")

print('SVM:\nr\r', classification_report(y_test, y_prediction))
print("SVM:정답율", ac_score)

from sklearn.metrics import roc_curve

probs = svm_model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, probs)

from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='(AUC = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC(Receiver Operating Characteristic) curve')
plt.legend(loc='best')
plt.show()
