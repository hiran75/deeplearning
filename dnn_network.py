
from keras.models import Sequential
from keras.layers import Dense

def make_network(features, hiddenlayer):
    # 모델 네트워크 구성
    if hiddenlayer == 2:
        model = Sequential()
        model.add(Dense(20, input_dim=features, activation='relu'))
        model.add(Dense(5, activation='relu'))
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

