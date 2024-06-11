from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# from darts.models import Prophet
from darts.models import XGBModel
from darts.models import CatBoostModel
from darts.models import NHiTSModel
from darts.models import NLinearModel
from darts.models import DLinearModel
import matplotlib.pyplot as plt
from darts import TimeSeries
import argparse
import os
import warnings
import time
import sys
import logging
from sklearn.preprocessing import LabelEncoder
import psutil
import copy

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings(action='ignore')
os.getcwd()


model_logger = logging.getLogger('ml_logger')
model_logger.setLevel(logging.INFO)
model_handler = logging.FileHandler('../log/train_predict.log')
model_logger.addHandler(model_handler)

# 학습 데이터 세팅
def df_preprocess(df:pd.DataFrame, train_start_date:str, pred_start_date:str, time_col:str, target_col:str, y_col:list):
    
    df['STOR_CD'] = df['STOR_CD'].astype(str)
    df['ITEM_CD'] = df['ITEM_CD'].astype(str)
    
    train_start_date = str(train_start_date)
    pred_start_date = str(pred_start_date)
    
    df = df[(df[time_col]>=train_start_date)&(df[time_col]<pred_start_date)]
    
    label_col = y_col.copy()
    label_col.remove(target_col)
    le = LabelEncoder()
    df[label_col] = df[label_col].apply(le.fit_transform)
    
    # darts Data Type 
    df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize(None)
    df[y_col]=df[y_col].astype(np.float32)    

    store_unique = df['STOR_CD'].unique()
    item_unique = df['ITEM_CD'].unique()
    store_item_list = []

    for store in store_unique :
        for item in item_unique :
            store_item_list.append([store,item])   
    return df, store_item_list

# 대상 매장 아이템 선정
def set_train_df(df:pd.DataFrame, store:str, item:str, time_split:int, target_col:str, time_col:str, y_col:str) : 
    each_train_df = df.loc[(df['STOR_CD']==store) & (df['ITEM_CD']==item)]

    empty_columns = each_train_df.columns[df.isnull().all()]
    each_train_df = each_train_df.drop(empty_columns, axis=1)
    
    unique_counts = each_train_df.nunique()
    cols_to_exclude = unique_counts[unique_counts == 1].index.tolist()
    if 'STOR_CD' in cols_to_exclude :          
        cols_to_exclude.remove('STOR_CD')
    if 'ITEM_CD' in cols_to_exclude :
        cols_to_exclude.remove('ITEM_CD')
    if target_col in cols_to_exclude :         
        cols_to_exclude.remove(target_col)
    each_df_filtered = each_train_df.drop(cols_to_exclude, axis=1)
    set1 = set(y_col)
    set2 = set(cols_to_exclude)
    filterd_y_col = list(set1 - set2)
    
    train_df = TimeSeries.from_dataframe(each_df_filtered, time_col, filterd_y_col)
    train, val = train_df.split_after(time_split)
    return train, val

# 모델 선택
def choose_model(model_name:str, lags:int, input_size:int, output_length:int):
    model_name = model_name.lower()
    # if model_name == 'prophet' :
    #     model = Prophet()
    if model_name == 'xgboost' :
        model = XGBModel(
            lags=lags, output_chunk_length=output_length
            )                      # gpu_id=0
    if model_name == 'catboost' :
        model = CatBoostModel(
            lags=lags, 
            output_chunk_length=output_length
            # iterations=10, 
            # logging_level='Silent'
            )  # task_type='GPU'   - 'Silent' - 'Verbose' - 'Info' - 'Debug'
    if model_name == 'nlinear' :
        model = NLinearModel(
            input_chunk_length = input_size,
            output_chunk_length = output_length,
            #  pl_trainer_kwargs = {"accelerator": "cpu"}
            # pl_trainer_kwargs = {"accelerator":"auto", "devices":"auto", "strategy":"auto"}
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": 1}  # "devices": 1, "auto_select_gpus": True
            )
    if model_name == 'dlinear' :
        model = DLinearModel(
            input_chunk_length = input_size,
            output_chunk_length = output_length,
            pl_trainer_kwargs = {"accelerator": "mps", "devices": -1}
            )
    if model_name =='nhits' :
        model = NHiTSModel(
            input_chunk_length = output_length,
            output_chunk_length = output_length,
            activation = activation,
            # n_epochs=100
            # pl_trainer_kwargs = {"accelerator": "mps", "devices": -1}
            )       # n_epochs = 1
    return model

# 학습 & 예측
def train_predict(df:pd.DataFrame, store_item_list:list, time_split:pd.Timestamp, pred_date:int, target_col:str ,model_name, lags, input_size, output_length) :
    total_results_df = pd.DataFrame
    # logging.basicConfig(filename=f'{path}/log/training.log', level=logging.CRITICAL, format='%(asctime)s %(message)s')       # 학습 로그
    for i, (store, item) in enumerate(store_item_list):
        train, val = set_train_df(df, store, item, time_split, target_col, time_col, y_col)
        print(f"STORE : {store}, ITEM : {item}")
        # model_logger.info(f'STORE : {store}, ITEM : {item}')

        model = choose_model(model_name, lags, input_size, output_length)  # 매번 모델 초기화

        # each_model = copy.deepcopy(model)           # 반복 할때 마다 자원반환을 위해

        # 학습
        # each_model.fit(train)     # 학습 & 검증
        model.fit(train)  # 학습 & 검증

        # 모델 파라미터 확인
        # params = model.get_parameter()
        # training_iterations = params['iterations']
        # learning_rate = params['learning_rate']

        # logging.info(f"Training iterations: {training_iterations}")
        # logging.info(f"Learning rate: {learning_rate}")

        # 모델 저장
        # `save_model()` and loaded using :func:`load_model()`. Default: ``False``.
        # model_nhits.save_model(f'...')

        # 예측
        # preds = each_model.predict(n=pred_date)
        preds = model.predict(n=pred_date)

        preds_tolist = np.concatenate(preds[target_col].values()).tolist()


        # 모델 할당 크기 확인
        # size_in_bytes = sys.getsizeof(each_model)
        # size_in_gb = size_in_bytes / (1024 ** 3)
        # print(f"Each model Memory usage: {size_in_gb:.2f} GB")

        # 모델 메모리 반환
        # del each_model

        # 메모리 확인
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        memory_gb = memory_bytes / (1024**3)
        print(f"Memory usage: {memory_gb:.2f} GB")

        # store, item 별 df (Features 중 target)
        each_result_df = pd.DataFrame(
            {"DT": preds.time_index.values, "STOR_CD": store, "ITEM_CD": item, "pred": preds_tolist}
        )

        # 후처리
        each_result_df["fillna_pred"] = each_result_df["pred"].diff().fillna(0)  # 하루 전 값 빼기
        each_result_df["fillna_pred"][0] = (each_result_df["pred"][0] - train[target_col].values()[-1])  # 예측 첫날값 - 실제 마지막날값
        each_result_df["zero_pred"] = each_result_df["fillna_pred"].apply(lambda x: 0 if x < 0 else x)  # 0 이하 0으로 치환
        each_result_df["round_pred"] = each_result_df["zero_pred"].round(0)  # 반올림

        # 전체 결과 df
        if i == 0:
            total_results_df = each_result_df
        else:
            total_results_df = pd.concat([total_results_df, each_result_df], ignore_index=True)
    return total_results_df

parser = argparse.ArgumentParser(description='darts')
parser.add_argument('--path', type=str, default='..')
parser.add_argument('--raw_data', type=str, default='2023-04-28_rawdatasets.csv')
parser.add_argument('--pred_date', type=int, default=7)
parser.add_argument('--train_start_date',type=str, default='2022-07-01')
parser.add_argument('--pred_start_date',type=str, default='2023-04-27')
args = parser.parse_args()

if __name__ == '__main__' :
    try:
        path = args.path
        df = pd.read_csv(f"{path}/datasets/{args.raw_data}", dtype={"STOR_CD": "str"})
        pred_date = args.pred_date  # 예측 기간
        train_start_date = args.train_start_date  # 학습 시작 일
        pred_start_date_ori = args.pred_start_date  # 예측 시작 일
        time_col = "DT"  # 시간 컬럼     STOCK_DT
        over_tr = True  # 3이상 실제값 2로 치환한 데이터(True), 안한 원본 누적(False)

        # 3이상 실제값 2로 치환한 데이터(True), 안한 원본 누적(False)
        if over_tr == True:
            over_flag = "_over"
            target_col = "GO_QTY_CS_OVER"
        else:
            over_flag = ""
            target_col = "GO_QTY_CS"

        y_col = ["WEEK_MONTH_STD", "D_SEQ", "WEEK", target_col, "holiday"]
        # y_col = ['GO_QTY','GO_QTY_CS','GO_QTY_OVER','GO_QTY_CS_OVER','DT',
        #          'YEAR','MONTH','DAY','WEEK_MONTH_STD','DOW',
        #          'DY','CK_31','holiday','YN','Week',
        #          'Deliverdow','deliverday','31day']            # y 컬럼 피처 추가
        # model_list = ['catboost','nhits', 'dlinear','nlinear']
        model_list = ["nlinear"]  # ['catboost','nhits', 'dlinear','nlinear']

        # 모델 하이퍼 파라미터
        input_size = 7  # input chunk 길이
        output_length = 14  # 예측 output chunk 길이 (ex. 30일 예측 output_length 14 --> 14일 + 14일 +1 일 = 30일) 한달 예측 시 14가 더 잘 나옴
        activation = "ReLU"  # ReLU, RReLU
        lags = 5  # catboost,xgboost param

        # 프로세스 시작 시간 기록
        process_start = time.strftime("%c", time.localtime(time.time()))
        print(f"Process Start Time : {process_start}")
        model_logger.info(f"Process Start Time : {process_start}")

        ## 예측일 추가 ##
        # ( 실행 날짜 - raw 데이터 마지막 날짜 ) 를 예측일에 더함.
        raw_last_date = str(df["DT"].unique()[-1])
        pred_start_date = pd.to_datetime(pred_start_date_ori)
        raw_last_date = pd.to_datetime(raw_last_date)

        if pred_start_date <= raw_last_date:  # 과거 시점을 예측해 테스트 하는 경우 (테스트용)
            print(f"Predict Start Date : {pred_start_date}")
            print(f"Predict Date : {pred_date} days)")
            model_logger.info(f"Predict Start Date : {pred_start_date}")
            model_logger.info(f"Predict Date : {pred_date} days)")

        if pred_start_date > raw_last_date:  # 미래 예측할 경우
            minus = pred_start_date - raw_last_date - timedelta(days=1)
            pred_start_date = raw_last_date + timedelta(days=1)  # 예측 시작일은 ( raw data 의 마지막 날짜 + 1 )이 됨
            pred_start_date = str(pred_start_date._date_repr)
            print(f"Today (Process Start date): {pred_start_date_ori}")
            print(f"Last Day of Row Data : {raw_last_date._date_repr}")
            print(f"Predict Start Date : {pred_start_date}")
            print(f"Predict Date : {pred_date+minus.days} ({pred_date}+{minus.days} days)")
            model_logger.info(f"Today (Process Start date): {pred_start_date_ori}")
            model_logger.info(f"Last Day of Row Data : {raw_last_date._date_repr}")
            model_logger.info(f"Predict Start Date : {pred_start_date}")
            model_logger.info(f"Predict Date : {pred_date+minus.days} ({pred_date}+{minus.days} days)")
            pred_date = pred_date + minus.days  # 예측기간은 ( 실행 날짜 - raw 데이터 마지막 날짜 ) 를 더한 기간

        # 저장 폴더 생성
        os.makedirs(f"{path}/results/{pred_start_date_ori}", exist_ok=True)  # exist_ok 존재할 경우 아무것도 안함
        os.makedirs(f"{path}/log", exist_ok=True)

        # 전처리
        # store, item 을 리스트화 해서 학습+예측 반복 하고자 함
        df, store_item_list = df_preprocess(df, train_start_date, pred_start_date, time_col, target_col, y_col)

        # 학습데이터 split
        # time_split = pd.Timestamp(pred_start_date) - datetime.timedelta(days=pred_date*n_val)
        time_split = pd.Timestamp(pred_start_date) - timedelta(days=1)

        #### 학습 & 추론 ###
        train_time = []
        for model_name in model_list:
            # print(f'============ {model_name} ============')
            # model_logger.info(f'============ {model_name} ============')

            # model_start = time.strftime('%c', time.localtime(time.time()))
            # print(f'{model_name} start time : {model_start}')
            # model_logger.info(f'{model_name} start time : {model_start}')

            # model = choose_model(model_name, lags, input_size, output_length)     # model 한번 초기화 이후 덮어 씌우면서 학습,예측 --> 매번 모델 초기화,학습,예측 으로 변경
            pred_result = train_predict(df, store_item_list, time_split, pred_date, target_col, model_name, lags, input_size, output_length)
            pred_result.to_csv(f"{path}/results/{pred_start_date_ori}/{pred_start_date_ori}_pred_raw.csv",index=False)
            # pred_result.to_csv(f'./results/{pred_start_date}/{model_name}_in_{input_size}_out_{output_length}_val_{n_val}_act_{activation}.csv', index=False)     # 하이퍼 파라미터를 결과 csv 파일 명으로 사용 / 모델 마다 사용 하이퍼 파라미터가 다름

            # # 모델 종료 시간
            # model_end = time.strftime('%c', time.localtime(time.time()))
            # print(f'{model_name} end time : {model_end}')
            # model_logger.info(f'{model_name} end time : {model_end}')

        # 프로세스 종료 시간 기록
        process_end = time.strftime("%c", time.localtime(time.time()))
        print(f"Process End Time : {process_end}")
        model_logger.info(f"Process End Time : {process_end}")
        model_logger.info(f"==========================================")

        # ### 검증 (Test code) ###
        # # 예측 실제값 전체 합계 비교, 최적 모델 찾기
        # act_df = pd.read_csv(f'{path}/datasets/20230414_rawdatasets.csv')         # 차후 검증 편리하게 하려고 실제값은 따로 불러옴

        # # 실제값 slicing
        # act_start_date = pd.Timestamp(pred_start_date) + datetime.timeelta(days=1)            # 학습 마지막 날짜 + 1일
        # act_end_date = pd.Timestamp(pred_start_date) + datetime.timedelta(days=pred_date)     # 학습 마지막 날짜 + 14일
        # time_col = 'STOCK_DT'
        # act_target = 'GO_QTY'
        # y_col = ['GO_QTY']

        # act_df, store_item_list = df_preprocess(act_df, act_start_date, act_end_date, time_col, act_target, y_col)
        # act_sum = act_df['GO_QTY'].sum()

        # print(f'Actual sum : {act_sum}')

        # score_list = []
        # # model_list = ['catboost','nlinear']
        # for i, model_name in enumerate(model_list) :
        #     print(f'============ {model_name} ============')
        #     pred_result= pd.read_csv(f'{path}/results/{pred_start_date}/{pred_start_date}_{model_name}{over_flag}.csv')
        #     # pred_result= pd.read_csv(f'./results/{pred_start_date}/{model_name}_in_{input_size}_out_{output_length}_val_{n_val}_act_{activation}.csv')
        #     score = scoring(pred_result, act_sum, model_name)
        #     score_list.append(score)

        # best_model = min(score_list, key=lambda x: x[3])
        # print('===============================')
        # print(f'Best model: {best_model[0]}')

    except Exception as e:
        model_logger.error(str(e), exc_info=True)
        
        process_start = time.strftime('%c', time.localtime(time.time()))
        print(f'[ ERROR ] Main Process')
        print(f'Process End time : {process_start}')
        model_logger.info(f'Process End time : {process_start}')
        model_logger.info(f'==========================================')
        raise SystemExit("")
        sys.exit()
