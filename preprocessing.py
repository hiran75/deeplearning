import numpy
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def pre_datamanager(filename, week):
    # seed 값 설정
    seed = 0
    numpy.random.seed(seed)
    tf.random.set_seed(seed)

    df_pre = pd.read_csv(filename,  header=0)  # CSV파일을 불러오는 함수를 이용

    # print(df_pre.info())
    features = len(df_pre.columns) - 1
    # 데이터 내부의 기호를 숫자로 변환하기--- (※2)

    df_pre = df_pre.sample(frac=1)

    # 학과코드를 onhot encoding 함##########################
    le = LabelEncoder()
    le.fit(df_pre['dept'])
    df_pre['dept'] = le.transform(df_pre['dept'])

    le.fit(df_pre['gender'])
    df_pre['gender'] = le.transform(df_pre['gender'])

    le.fit(df_pre['area'])
    df_pre['area'] = le.transform(df_pre['area'])

    # 재학생 데이터만 활용
    #df_pre = df_pre[df_pre['entYn'] == 0]

    # 재학생 데이터만 활용
    if week!=0:
        df_pre = df_pre[df_pre['weekseq'] == week]

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
        ##################over sampling end b##################

    return X_train, X_test, y_train, y_test, features


def pre_datamanager_noheader(filename, oversample_flage):
    # seed 값 설정
    seed = 0
    numpy.random.seed(seed)
    tf.random.set_seed(seed)

    df_pre = pd.read_csv(filename, header=None)

    #df_pre.dropna()
    # print(df_pre.info())
    features = len(df_pre.columns) - 1
    # 데이터 내부의 기호를 숫자로 변환하기--- (※2)

    df_pre = df_pre.sample(frac=1)


    # 학과코드를 onhot encoding 함##########################
    #le = LabelEncoder()
    #le.fit(df_pre[0])
    #df_pre[0] = le.transform(df_pre[0])
    # 학과코드를 onhot encoding end ##########################
    #df_pre

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

    oversample_flage= False
    if oversample_flage:
        smote = SMOTE(random_state=0)
        X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
        X_train = X_train_over
        y_train = y_train_over

    return X_train, X_test, y_train, y_test, features


