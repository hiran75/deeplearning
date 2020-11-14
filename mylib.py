import os

def setting(model_name):
    # 모델 저장 폴더 설정
    MODEL_DIR = './model/' + model_name
    #print(MODEL_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    return 1

MODEL_DIR=""


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



from keras.models import Sequential
from keras.layers import Dense

def make_network(features, hiddenlayer):
    # 모델 네트워크 구성
    if hiddenlayer == 2:
        model = Sequential()
        model.add(Dense(30, input_dim=features, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))


    if hiddenlayer == 3:
        model = Sequential()
        model.add(Dense(28, input_dim=features, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    if hiddenlayer == 4:
        model = Sequential()
        model.add(Dense(28, input_dim=features, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    if hiddenlayer == 5:
        model = Sequential()
        model.add(Dense(28, input_dim=features, activation='relu'))
        model.add(Dense(22, activation='relu'))
        model.add(Dense(18, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    if hiddenlayer == 6:
        model = Sequential()
        model.add(Dense(30, input_dim=features, activation='relu'))
        model.add(Dense(28, input_dim=features, activation='relu'))
        model.add(Dense(20, input_dim=features, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        #model.add(Dense(1, activation='softmax'))

    return model


# pandas 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class visualizer:
    def visualizer_main(df):
        colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
        plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다.

        # 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
        sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
        plt.show()

        grid = sns.FacetGrid(df, col='class')
        grid.map(plt.hist, 'plasma',  bins=10)
        plt.show()

        def fit(model, X_train, y_train, model_name, v_epoches, v_batch_size):

            MODEL_DIR = './model/' + model_name
            # 모델 저장 조건 설정
            modelpath = MODEL_DIR + "/{epoch:02d}-{val_loss:.4f}.hdf5"
            checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

            # 학습 자동 중단 설정
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

            # 데이터 학습
            history = model.fit(X_train, y_train, validation_split=0.20, epochs=v_epoches, batch_size=v_batch_size,
                                callbacks=[early_stopping_callback, checkpointer])

            y_vloss = history.history['val_loss']
            y_acc = history.history['acc']

            # x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
            x_len = numpy.arange(len(y_acc))
            plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
            plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

            plt.show()

            return model

        def model_evaluate(model, model_name, X_test, y_test, features, v_epoches, v_batch_size):
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
            print("/n #accuracy, precision, recall, f1_score")
            print(" # %.4f, %.4f, %4f, %.4f" % (accuracy, precision, recall, f1_socre))
            return accuracy

        def predict(model, X_test, y_test):
            # 예측 값과 실제 값의 비교
            Y_prediction = model.predict(X_test)
            for i in range(10):
                label = y_test[i]
                prediction = Y_prediction[i]
                if prediction > 0.5:
                    pre_label = 1
                else:
                    pre_label = 0
                print("실제: ", label, "예상", pre_label)


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
plt.title('ROC(Receiver Operating Characteristic) curve')
plt.legend(loc='best')
plt.show()


#네트워크 생성
model = network.make_network(features, v_layer)

# 모델 컴파일
model.compile(loss=cost_f,
              optimizer='adam',
              # metrics=['accuracy'])
              metrics=['accuracy', lib.recall_m, lib.precision_m, lib.f1_m])


from sklearn.preprocessing import LabelEncoder



# 학과코드를 onhot encoding 함##########################
le = LabelEncoder()
le.fit(df_pre['dept'])
df_pre['dept'] = le.transform(df_pre['dept'])
# 학과코드를 onhot encoding end ##########################
