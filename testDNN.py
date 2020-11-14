
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense


v_layer = 3
cost_f = "binary_crossentropy"
v_epoches = 500
v_batch_size = 20

##############################################################

v_data = '20201114_v1'  # 사용할 데이터 셋

filename = "./dataset/" + v_data + ".csv"
model_name = v_data

import os

def setting(model_name):
    # 모델 저장 폴더 설정
    MODEL_DIR = './model/' + model_name
    #print(MODEL_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    return 1


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
df_pre = df_pre[df_pre['weekseq'] == 1]


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

# 데이터 정규화
X_train, X_test = X_train / 255, X_test / 255

# over sampling
from imblearn.over_sampling import SMOTE

oversample_flage = True
if oversample_flage:
    smote = SMOTE(random_state=0)
    X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
    X_train = X_train_over
    y_train = y_train_over

# 데이터 학습시키기 --- (※4)

# X_train, X_test, y_train, y_test, features= lib.pre_datamanager_noheader(filename, False)
# 네트워크 생성
model = Sequential()
model.add(Dense(50, input_dim=features, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# 모델 컴파일
model.compile(loss=cost_f,
              optimizer='adam',
              # metrics=['accuracy'])
              metrics=['accuracy', recall_m, precision_m, f1_m])

MODEL_DIR = './model/' + model_name
# 모델 저장 조건 설정
modelpath = MODEL_DIR + "/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

# 데이터 학습
history = model.fit(X_train, y_train, validation_split=0.20, epochs=v_epoches, batch_size=v_batch_size,
                    callbacks=[early_stopping_callback, checkpointer])


#y_vloss = history.history['val_loss']
#y_acc = history.history['acc']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
#x_len = numpy.arange(len(y_acc))
#plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
#plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

#plt.show()

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

import datetime

print('============================')
print("파일명 ", model_name)
print("요소갯수 ", features)
print("================================== ")

# 결과 출력
# print("\n epoches", v_epoches, "bat_size=", v_batch_size)
# print("\n 학습중단 + 모델 성능개선 : arly_stopping_callback:")
# print("\n 예측정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 테스트 데이터 검증
loss, accuracy, recall, precision, f1_socre = model.evaluate(X_test, y_test)
# accuracy  = model.evaluate(X_test, Y_test)
print('DNN_', datetime.datetime.now())
print("================================== ")
print("파일명 ", filename)
print("요소갯수 ", features)
print("================================== ")
print("정답률 =", accuracy)

print("/n #accuracy, precision, recall, f1_score")
print(" # %.4f, %.4f, %4f, %.4f" % (accuracy, precision, recall, f1_socre))

print("#file:", filename, "\n\n model:", model_name, "\n accuracy:", accuracy)

from sklearn.metrics import roc_curve

probs = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, probs)

from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='(AUC = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('DNN Model ROC(Receiver Operating Characteristic) curve')
plt.legend(loc='best')
plt.show()