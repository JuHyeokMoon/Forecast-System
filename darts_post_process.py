import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import datetime
import argparse
import sys
import logging
import time
import pymysql
import configparser as parsers

if not os.path.exists('/Users/moonju/Documents/darts/log/post_process.log'):
    open('/Users/moonju/Documents/darts/log/post_process.log', 'w').close()
    
post_logger = logging.getLogger('post_logger')
post_logger.setLevel(logging.INFO)
model_handler = logging.FileHandler('/Users/moonju/Documents/darts/log/post_process.log')
post_logger.addHandler(model_handler)

# 날짜 변수
def date_feature(df,store_type):
    for i,store in enumerate(df['STOR_CD'].unique()):
        each_store = df.loc[df['STOR_CD']==store]
        each_store['D_TYPE'] = store_type[store]
        if i == 0 :
            results = each_store
        else :
            results = pd.concat([results,each_store])
    return results

# 예측치 결합 방식
def forecast_result_sum(df:pd.DataFrame, round_type:str, round=1):
    round2_df = pd.DataFrame() 
    round1_df = pd.DataFrame() 
    for key,group in df.groupby(['STOR_CD','ITEM_CD']):
        if group['D_TYPE'].unique()[0] == '1101010':
            stock_of_mon = list(group.loc[group.DOW == 2]['DATE'])
            stock_of_tue = list(group.loc[group.DOW == 3]['DATE'])
            stock_of_thu = list(group.loc[group.DOW == 5]['DATE'])
            stock_of_sat = list(group.loc[group.DOW == 7]['DATE'])
            for idx in range(len(stock_of_mon)):
                tmp_df = group.loc[group.DATE == stock_of_mon[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE == stock_of_mon[idx]]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE == stock_of_mon[idx]]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE == stock_of_mon[idx]]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_tue)):
                tmp_df = group.loc[group.DATE == stock_of_tue[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_tue[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_tue[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_tue[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_thu)):
                tmp_df = group.loc[group.DATE == stock_of_thu[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_thu[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_thu[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_thu[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_sat)):
                tmp_df = group.loc[group.DATE == stock_of_sat[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_sat[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
        elif group['D_TYPE'].unique()[0] == '1010100':
            stock_of_mon = list(group.loc[group.DOW == 2]['DATE'])
            stock_of_wed = list(group.loc[group.DOW == 4]['DATE'])
            stock_of_fri = list(group.loc[group.DOW == 6]['DATE'])
    
            for idx in range(len(stock_of_mon)):
                tmp_df = group.loc[group.DATE == stock_of_mon[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_mon[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_mon[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_mon[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_wed)):
                tmp_df = group.loc[group.DATE == stock_of_wed[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_wed[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_wed[idx]][:2]['ACTUAL'].sum() 
                    tmp_pred = group.loc[group.DATE >= stock_of_wed[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_thu)):
                tmp_df = group.loc[group.DATE == stock_of_fri[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_fri[idx]][:3]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_fri[idx]][:3]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_fri[idx]][:3]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
        elif group['D_TYPE'].unique()[0] == '0101010':
            stock_of_tue = list(group.loc[group.DOW == 3]['DATE'])
            stock_of_thu = list(group.loc[group.DOW == 5]['DATE'])
            stock_of_sat = list(group.loc[group.DOW == 7]['DATE'])
    
            for idx in range(len(stock_of_tue)):
                tmp_df = group.loc[group.DATE == stock_of_tue[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_tue[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_tue[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_tue[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_thu)):
                tmp_df = group.loc[group.DATE == stock_of_thu[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_thu[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_thu[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_thu[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_sat)):
                tmp_df = group.loc[group.DATE == stock_of_sat[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:3]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_sat[idx]][:3]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:3]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
        elif group['D_TYPE'].unique()[0] == '1010110':
            stock_of_mon = list(group.loc[group.DOW == 2]['DATE'])
            stock_of_wed = list(group.loc[group.DOW == 4]['DATE'])
            stock_of_fri = list(group.loc[group.DOW == 6]['DATE'])
            stock_of_sat = list(group.loc[group.DOW == 7]['DATE'])
    
            for idx in range(len(stock_of_mon)):
                tmp_df = group.loc[group.DATE == stock_of_mon[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_mon[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_mon[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_mon[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_wed)):
                tmp_df = group.loc[group.DATE == stock_of_wed[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_wed[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_wed[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_wed[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_fri)):
                tmp_df = group.loc[group.DATE == stock_of_fri[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE == stock_of_fri[idx]]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE == stock_of_fri[idx]]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE == stock_of_fri[idx]]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
            for idx in range(len(stock_of_sat)):
                tmp_df = group.loc[group.DATE == stock_of_sat[idx]]
                if round_type == 'max':
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:2]['PRED'].max()
                if round_type == 'sum':
                    tmp_actual = group.loc[group.DATE >= stock_of_sat[idx]][:2]['ACTUAL'].sum()
                    tmp_pred = group.loc[group.DATE >= stock_of_sat[idx]][:2]['PRED'].sum()
                    tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round1_df = pd.concat([round1_df,tmp_df])
    
    round1_df = round1_df.sort_values(['STOR_CD','ITEM_CD','DATE'])

    if round == 1 :
        final_df = round1_df.copy()
    elif round == 2 : 
        for key,group in round1_df.groupby(['STOR_CD','ITEM_CD']):
            # for idx in range(0,len(group),2):
            #     tmp_actual = group.iloc[idx:idx+2]['ACTUAL']
            #     tmp_pred = group.iloc[idx:idx+2]['PRED']
            #     tmp_df = group.iloc[idx:idx+2][-1:]
            #     tmp_df['ACTUAL'] = tmp_actual
            #     tmp_df['PRED'] = tmp_pred
            #     round2_df = pd.concat([round2_df,tmp_df])
            
            for idx in range(0,len(group),2):
                tmp_actual = group.iloc[idx:idx+2]['ACTUAL'].sum()
                tmp_pred = group.iloc[idx:idx+2]['PRED'].sum()
                tmp_df = group.iloc[[idx]]
                tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round2_df = pd.concat([round2_df,tmp_df])
                
        final_df = round2_df.copy()
    
    final_df=final_df.sort_values(['STOR_CD','ITEM_CD','DATE'])
    
    # final_df = final_df[['DATE','STOR_CD','ITEM_CD','ACTUAL','PRED']]
    final_df = final_df[['DATE','STOR_CD','ITEM_CD','PRED']]
    
    return final_df

def server_connect(host,port,user,password,db):
    
    conn = pymysql.connect(host=host, 
                           user=user, 
                           port=port, 
                           password=password, 
                           db=db, 
                           charset='utf8', use_unicode=True,
                           cursorclass=pymysql.cursors.DictCursor)
    return conn

def get_calender_data(host,port,user,password,db):
    query = """
        select DT, YEAR,MONTH,DAY,WEEK_MONTH_STD,DOW ,date_format(str_to_date(dt , '%Y%m%d') , '%a') as DY
        ,case when  substr(month_last_dt, 7,2) = '31' then 1 else 0 end CK_31
        from mst_clndr
        where dt >= '20210101';
    """
    conn = server_connect(host,port,user,password,db)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()

    calender_df = pd.DataFrame(result)
    calender_df = calender_df.loc[calender_df.DT <= '20331231'] 
    calender_df['DT'] = pd.to_datetime(calender_df['DT'],format="%Y%m%d")

    return calender_df

parser = argparse.ArgumentParser(description='darts')

parser.add_argument('--path', type=str, default='/Users/moonju/Documents/darts')
parser.add_argument('--pred_start_date',type=str, default='2023-05-01')
parser.add_argument('--pred_data', type=str, default='test_pred_raw.csv')
parser.add_argument('--actual_data', type=str, default='2023-05-19_rawdatasets.csv')

args = parser.parse_args()



if __name__ == '__main__' :
    try :
        process_start = time.strftime('%c', time.localtime(time.time()))
        print(f'Process Start Time : {process_start}')
        post_logger.info(f'Process Start Time : {process_start}')

        path = args.path
        pred_start_date = args.pred_start_date
        pred_data = args.pred_data
        actual_data = args.actual_data

        model = 'nlinear'
        over_tr = True

        if over_tr == True :
            over_flag = '_over'
            target_col = 'GO_QTY_CS_OVER'
        else :
            over_flag = ''
            target_col = 'GO_QTY_CS'

        model = model.lower()
        # print(f'Model : {model}')
        # print(f'Over : {target_col}')
        
        actual_df = pd.read_csv(f"{path}/datasets/{actual_data}")
        store_df = pd.read_csv(f"{path}/datasets/mst_stor.csv")
        item_df = pd.read_csv(f"{path}/datasets/mst_item.csv")
        
        actual_df['DELIVERDOW'] = actual_df['DELIVERDOW'].astype(str)
        actual_df['STOR_CD'] = actual_df['STOR_CD'].astype(str)

        actual_df['DELIVERDOW'] = actual_df['DELIVERDOW'].apply(lambda x: '0101010' if x == '101010' else x)
        
        df = pd.read_csv(f"{path}/results/{pred_start_date}/{pred_data}")
        df['STOR_CD'] = df['STOR_CD'].astype(str)
        df['DT'] = pd.to_datetime(df['DT'],format="%Y-%m-%d")

        properties = parsers.ConfigParser()
        properties.read(f"{path}/code/config.ini")
        #! DB 변경으로 해당 부분 수정
        host = str(properties['sbp']['host'])
        port = int(properties['sbp']['port'])
        user = str(properties['sbp']['user'])
        password = str(properties['sbp']['password'])
        db = str(properties['sbp']['db'])

        cal = get_calender_data(host=host, port=port, user=user,password=password, db=db)
        df = pd.merge(left=df,right=cal,how='left',on='DT')
        df['STOR_CD'] = df['STOR_CD'].astype(str)
        df['DOW'] = df['DOW'].astype(int)

        # 사전 생성
        store_dict = dict(zip(store_df['STOR_NM'], store_df['STOR_CD']))
        item_dict = dict(zip(item_df['ITEM_NM'], item_df['ITEM_CD']))
        store_type = dict(zip(actual_df['STOR_CD'], actual_df['DELIVERDOW']))

        # Dataframe pre-process
        df.columns = [ col.upper() for col in df.columns ]
        df = df.reset_index(drop=True)
        df['DT'] = pd.to_datetime(df['DT'])

        df = date_feature(df,store_type=store_type)

        df = df.rename(columns={'DT':'DATE'})
        df = df.rename(columns={'GO_QTY':'ACTUAL'})
        
        merged_normal = df[['DATE', 'STOR_CD', 'ITEM_CD','ROUND_PRED', 'D_TYPE', 'DOW']]
        merged_normal = merged_normal.rename(columns={'ROUND_PRED':'PRED'})
        round_1_max_after_postproc_df = forecast_result_sum(df=merged_normal,round_type='max',round=1)  # round (deafult) : 1
        round_1_max_after_postproc_df.to_csv(f"{path}/results/{pred_start_date}/{pred_start_date}_results.csv", index=False)
        
        print(f'Post Process Done. Saved {pred_start_date}_results.csv')
        post_logger.info(f'Post Process Done. Saved {pred_start_date}_results.csv')
        process_end = time.strftime('%c', time.localtime(time.time()))
        print(f'Process End Time : {process_end}')
        post_logger.info(f'Process End Time : {process_end}')
        post_logger.info(f'==========================================')

    except Exception as e:
        post_logger.error(str(e), exc_info=True)
        
        process_end = time.strftime('%c', time.localtime(time.time()))
        print(f'[ ERROR ] Post Process.')
        print(f'Process End time : {process_end}')
        post_logger.info(f'Process End Time : {process_end}')
        post_logger.info(f'==========================================')
        raise SystemExit("")
        sys.exit()
