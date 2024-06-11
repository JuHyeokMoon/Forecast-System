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

if not os.path.exists('../log/verification.log'):
    open('../log/verification.log', 'w').close()

ver_logger = logging.getLogger('ver_logger')
ver_logger.setLevel(logging.INFO)
model_handler = logging.FileHandler('../log/verification.log')
ver_logger.addHandler(model_handler)


def date_feature(df,store_type):
    for i,store in enumerate(df['STOR_CD'].unique()):
        each_store = df.loc[df['STOR_CD']==store]
        each_store['D_TYPE'] = store_type[store]
        if i == 0 :
            results = each_store
        else :
            results = pd.concat([results,each_store])

    return results


def fill_date(store):
    filled_df = pd.DataFrame(columns=['STOR_CD','ITEM_CD','GO_QTY',"GO_QTY_CS","GO_QTY_OVER","GO_QTY_CS_OVER"])

    for key,group in store.groupby(['STOR_CD','ITEM_CD']):
        group['DT']=pd.to_datetime(group['DT'],infer_datetime_format=True,format='%Y%m%d')
        all_days = pd.date_range(group.DT.min(),datetime.date.today(),freq='D')
        group.set_index(['DT'],drop=True,inplace=True)
        group = group.reindex(all_days,fill_value=np.nan)
        missing_fill_value = {
            'STOR_CD':group.STOR_CD.unique()[0],
            'ITEM_CD':group.ITEM_CD.unique()[0],
            'GO_QTY':0,
            "GO_QTY_CS":group.GO_QTY_CS.ffill(),
            "GO_QTY_OVER":0,
            "GO_QTY_CS_OVER":group.GO_QTY_CS_OVER.ffill()
        }
        group = group.fillna(missing_fill_value)
        filled_df = pd.concat([filled_df,group])
    filled_df = filled_df.reset_index().rename(columns={'index':'DT'})
    return filled_df


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
        for _,group in round1_df.groupby(['STOR_CD','ITEM_CD']):
            
            for idx in range(0,len(group),2):
                tmp_actual = group.iloc[idx:idx+2]['ACTUAL'].sum()
                tmp_pred = group.iloc[idx:idx+2]['PRED'].sum()
                tmp_df = group.iloc[[idx]]
                tmp_df['ACTUAL'] = tmp_actual
                tmp_df['PRED'] = tmp_pred
                round2_df = pd.concat([round2_df,tmp_df])
                
        final_df = round2_df.copy()
    
    final_df=final_df.sort_values(['STOR_CD','ITEM_CD','DATE'])
    
    final_df = final_df[['DATE','STOR_CD','ITEM_CD','ACTUAL','PRED']]
    
    return final_df


def post_process(df:pd.DataFrame, actual_df:pd.DataFrame, target_col:str) :
    flag_df = df.rename(columns={'PRED':'ORI_PRED'})

    for j, store in enumerate(flag_df['STOR_CD'].unique()) :
        for i, item in enumerate(flag_df['ITEM_CD'].unique()) :
            
            each_df = flag_df.loc[(flag_df['STOR_CD']== store) & (flag_df['ITEM_CD']== item)]

            compare_date = pd.Timestamp(each_df['DATE'].values[1]) - datetime.timedelta(days=1)  

            each_df['FILL_NA_PRED'] = each_df['ORI_PRED'].diff().fillna(0)                          
            each_df['FILL_NA_PRED'][0] = each_df['ORI_PRED'].values[0] - actual_df.loc[actual_df['DATE'] == compare_date, target_col].values[0]   
            each_df['ZERO_PRED'] = each_df['FILL_NA_PRED'].apply(lambda x : 0 if x < 0 else x)      
            each_df['PRED'] = each_df['ZERO_PRED'].round(0)                                         
            
            if i == 0 :
                each_results = each_df
            else : 
                each_results = pd.concat([each_results, each_df])

        if j == 0 :
            results = each_results
        else : 
            results = pd.concat([results, each_results])
    return results


def make_gap_table(case_1_df:pd.DataFrame, case_2_df:pd.DataFrame, case_3_df:pd.DataFrame):
    
    df_list = [case_1_df, case_2_df, case_3_df]
    
    for i, df in enumerate(df_list) :
        
        df['ACTUAL'] = df['ACTUAL'].astype(int)
        df['PRED'] = df['PRED'].astype(int)
        diff_list = df['ACTUAL'] - df['PRED']         
        df['GAP'] = diff_list
        col_name = 'CASE_' + str(i+1) + '_GAP'
        df = df.rename(columns={'GAP': col_name})
        
        if i == 0 :
            results = df
        else :
            each_gap_col = df[col_name]
            results=pd.concat([results,each_gap_col],axis=1)
    return results


parser = argparse.ArgumentParser(description='darts')

parser.add_argument('--path', type=str, default='.')
parser.add_argument('--pred_start_date',type=str, default='2023-03-01')
parser.add_argument('--pred_data', type=str, default='2023-03-01_nlinear_over.csv')
parser.add_argument('--actual_data', type=str, default='20230414_rawdatasets.csv')

args = parser.parse_args()


if __name__ == '__main__' :

    try:
        process_start = time.strftime('%c', time.localtime(time.time()))
        print(f'Process Start Time : {process_start}')
        ver_logger.info(f'Process Start Time : {process_start}')

        path = args.path
        pred_start_date = args.pred_start_date
        pred_data = args.pred_data
        actual_data = args.actual_data   

        over_tr = True

        if over_tr == True :
            over_flag = '_over'
            target_col = 'GO_QTY_CS_OVER'
        else :
            over_flag = ''
            target_col = 'GO_QTY_CS'

        print(f'Over : {target_col}')

        df = pd.read_csv(f"{path}/results/{pred_start_date}/{pred_data}")
        df = df.rename(columns={'DT':'DATE'})
        df['STOR_CD'] = df['STOR_CD'].astype(str)

        actual_df = pd.read_csv(f"{path}/datasets/{actual_data}")
        store_df = pd.read_csv(f"{path}/datasets/mst_stor.csv")
        item_df = pd.read_csv(f"{path}/datasets/mst_item.csv")

        actual_df['DELIVERDOW'] = actual_df['DELIVERDOW'].astype(str)
        actual_df['STOR_CD'] = actual_df['STOR_CD'].astype(str)
        actual_df['DELIVERDOW'] = actual_df['DELIVERDOW'].apply(lambda x: '0101010' if x == '101010' else x)

        store_dict = dict(zip(store_df['STOR_NM'], store_df['STOR_CD']))
        item_dict = dict(zip(item_df['ITEM_NM'], item_df['ITEM_CD']))
        store_type = dict(zip(actual_df['STOR_CD'],actual_df['DELIVERDOW']))

        df.columns = [ col.upper() for col in df.columns ] 
        df = df.reset_index(drop=True)
        df['DATE'] = pd.to_datetime(df['DATE']) 
        df = date_feature(df,store_type=store_type)

        actual_df = fill_date(actual_df)
        actual_df['DT'] = pd.to_datetime(actual_df['DT'],format="%Y%m%d")
        actual_df = actual_df.rename(columns={'DT':'DATE'})
        merged_df = pd.merge(left = df,right = actual_df, how='left',on=['STOR_CD','ITEM_CD','DATE'])
        
        merged_df = merged_df.rename(columns={'GO_QTY':'ACTUAL'})
        merged_df['YEAR'] = merged_df['YEAR'].astype(int)
        merged_df['MONTH'] = merged_df['MONTH'].astype(int)
        merged_df['DAY'] = merged_df['DAY'].astype(int)
        merged_df['DOW'] = merged_df['DOW'].astype(int)

        ori_pred_df = merged_df[['DATE', 'STOR_CD', 'ITEM_CD', 'PRED', 'FILLNA_PRED', 
                                 'ZERO_PRED','ROUND_PRED', 'ACTUAL']]      

        merged_normal = merged_df[['DATE', 'STOR_CD', 'ITEM_CD', 'ROUND_PRED', 'D_TYPE', 
                                   'ACTUAL', 'GO_QTY_CS', 'GO_QTY_OVER', 'GO_QTY_CS_OVER', 'YEAR', 
                                   'MONTH', 'DAY', 'WEEK_MONTH_STD', 'DOW', 'CK_31', 
                                   'holiday', 'YN', 'WEEK', 'DELIVERDOW']]
        
        merged_normal = merged_normal.rename(columns={'ROUND_PRED':'PRED'})

        merged_max = merged_df[['DATE', 'STOR_CD', 'ITEM_CD', 'PRED', 'D_TYPE', 
                                'ACTUAL', 'GO_QTY_CS', 'GO_QTY_OVER', 'GO_QTY_CS_OVER', 'YEAR', 
                                'MONTH', 'DAY', 'WEEK_MONTH_STD', 'DOW', 'CK_31', 
                                'holiday', 'YN', 'WEEK', 'DELIVERDOW']]
        
        round_1_df = forecast_result_sum(df=merged_normal,round_type='sum',round=1)  
        round_2_df = forecast_result_sum(df=merged_normal,round_type='sum',round=2)
        round_1_max_after_postproc_df = forecast_result_sum(df=merged_normal,round_type='sum',round=1)  # round (deafult) : 1
        round_1_max_after_postproc_df['ACTUAL'] = round_1_max_after_postproc_df['ACTUAL'].astype(int)
        round_1_max_after_postproc_df['PRED'] = round_1_max_after_postproc_df['PRED'].astype(int)
        diff_list = round_1_max_after_postproc_df['ACTUAL'] - round_1_max_after_postproc_df['PRED']           
        round_1_max_after_postproc_df['GAP'] = diff_list
        
        print('\n########### Total ###########')
        ver_logger.info(f'\n########### Total ###########')
        score_df = round_1_max_after_postproc_df.copy()
        range_dic = {}
        for range in score_df['GAP'].unique() :
            each_range_df = score_df['GAP'].loc[score_df['GAP']==range]
            range_count = len(each_range_df)
            range_dic.setdefault(range,range_count)

        total_count = len(score_df)
        per_0 = round((range_dic[0]/total_count)*100,2)
        per_1 = round(((range_dic.get(0, 0) + range_dic.get(1, 0) + range_dic.get(-1, 0))/total_count)*100,2)
        store_count = len(score_df['STOR_CD'].unique())
        item_count = len(score_df['ITEM_CD'].unique())
        
        print(f'Total')
        print(f'STORE : {store_count}, ITEM : {item_count}')
        print(f'GAP 0 (%) : {per_0} %')
        print(f'GAP -1 ~ 1 (%) : {per_1} %')
        ver_logger.info(f'GAP 0 (%) : {per_0} %')
        ver_logger.info(f'GAP -1 ~ 1 (%) : {per_1} %')

        print('\n########### Store ###########')
        ver_logger.info(f'\n########### Store ###########')
        for i, store_cd in enumerate(score_df['STOR_CD'].unique()) :
            range_dic = {}
            each_store_df = score_df.loc[score_df['STOR_CD']==store_cd]
            for range in each_store_df['GAP'].unique() :
                each_range_df = each_store_df['GAP'].loc[each_store_df['GAP']==range]
                range_count = len(each_range_df)
                range_dic.setdefault(range,range_count)
            
            total_count = len(each_store_df)
            per_0 = round((range_dic[0]/total_count)*100,2)
            per_1 = round(((range_dic.get(0, 0) + range_dic.get(1, 0) + range_dic.get(-1, 0))/total_count)*100,2)     
            
            print(f'------- {store_cd} -------')
            print(f'GAP 0 (%) : {per_0} %')
            print(f'GAP -1 ~ 1 (%) : {per_1} %')
            ver_logger.info(f'------- {store_cd} -------')
            ver_logger.info(f'GAP 0 (%) : {per_0} %')
            ver_logger.info(f'GAP -1 ~ 1 (%) : {per_1} %')

        print('\n########### ITEM ###########')
        ver_logger.info(f'\n########### ITEM ###########')
        for i, item_cd in enumerate(score_df['ITEM_CD'].unique()) :
            range_dic = {}
            each_item_df = score_df.loc[score_df['ITEM_CD']==item_cd]
            for range in each_item_df['GAP'].unique() :
                each_range_df = each_item_df['GAP'].loc[each_item_df['GAP']==range]
                range_count = len(each_range_df)
                range_dic.setdefault(range,range_count)
            
            total_count = len(each_item_df)
            per_0 = round((range_dic[0]/total_count)*100,2)
            per_1 = round(((range_dic.get(0, 0) + range_dic.get(1, 0) + range_dic.get(-1, 0))/total_count)*100,2)      
            
            print(f'------- {item_cd} -------')
            print(f'GAP 0 (%) : {per_0} %')
            print(f'GAP -1 ~ 1 (%) : {per_1} %')
            ver_logger.info(f'------- {item_cd} -------')
            ver_logger.info(f'GAP 0 (%) : {per_0} %')
            ver_logger.info(f'GAP -1 ~ 1 (%) : {per_1} %')
        
        round_1_max_after_postproc_df.to_csv(f"{path}/results/{pred_start_date}/{pred_start_date}_gap.csv", index=False)
        
        process_end = time.strftime('%c', time.localtime(time.time()))
        print(f'Process End Time : {process_end}')
        ver_logger.info(f'Process End Time : {process_end}')
        ver_logger.info(f'==========================================')
    
    except Exception as e:
        ver_logger.error(str(e), exc_info=True)

        process_end = time.strftime('%c', time.localtime(time.time()))
        print(f'[ ERROR ] Verification.')
        print(f'Process End Time : {process_end}')
        ver_logger.info(f'Process End Time : {process_end}')
        ver_logger.info(f'==========================================')
        raise SystemExit("")
        sys.exit()

    