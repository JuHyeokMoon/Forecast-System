from cgitb import handler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta

import requests
import datetime
from bs4 import BeautifulSoup

import pymysql
import argparse
import os

from pathlib import Path

import logging
import logging.handlers
import sys
from rich.logging import RichHandler

import configparser as parsers

def print_whichday(year, month, day) :
    r = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    aday = datetime.date(year, month, day)
    bday = aday.weekday()
    return r[bday]

def get_request_query(url, operation, params, serviceKey):
    import urllib.parse as urlparse
    params = urlparse.urlencode(params)
    request_query = url + '/' + operation + '?' + params + '&' + 'serviceKey' + '=' + serviceKey
    return request_query

def server_connect(host,port,user,password,db):
    conn = pymysql.connect(host=host, # host 주소 입력 ip주소
                           user=user, # db에 접근할 user id
                           port=port, # host의 port번호
                           password=password, # 비밀번호
                           db=db, # 접속할 DB이름
                           charset='utf8', use_unicode=True,
                           cursorclass=pymysql.cursors.DictCursor)
    return conn

# 직접 작성하여 사용하다가 이후 사장된 코드
def get_raw_data(host,port,user,password,db):
    raw_data_query = """
        select STOR_CD, ITEM_CD ,STOCK_DT, sum(go_qty) as GO_QTY, sum(GO_QTY) over(PARTITION BY STOR_CD,ITEM_CD ORDER BY STOCK_DT) as GO_QTY_CS,
        if(GO_QTY>=3, 2,GO_QTY) as GO_QTY_OVER,
        sum(if(GO_QTY>=3,2,GO_QTY)) OVER(PARTITION BY STOR_CD,ITEM_CD ORDER BY STOCK_DT) as GO_QTY_CS_OVER
        from
        (	
            select  ADJ_QTY ,STOCK_DT ,STOR_CD   ,GO_QTY ,
            case when ITEM_CD = 'A50685' then 'A56077'
            when ITEM_CD = 'A59960' then 'A56076' else ITEM_CD end ITEM_CD
            from ###
            where cmp_cd = 'BRKR' and stock_dt >= '20220101'
            and item_cd in
            ('A51940','A56035','A50111','A50554','A50842','A56076','A50410','A50607','A59964','A56077',
            'A55105','A51872','A50382','A52013','A50649','A50438','A50135','A50685','A59960')
            and stor_cd in 
            (
                '62070','22832','2205E','32981','2264H','11227','11233','11445','11459','11465',
                '11862','11991','110B1','11562','31781','31970','12782','11112','62019','72095',
                '114B1','11661','31772','31780','31861','32986','21480','11651','42973','72051',
                '621DC','62066','21673','22437','620A2','21729','211B1','11581','72173','11891',
                '72011','42963','22998','12A37','11235','61190','21551','42871','2221B','11467',
                '32990','117B3','72110','72192','62049','12161','62037','72151','62142','2257C',
                '329BG','42982','11863'
            )
            order by STOCK_DT asc
        )a
        where go_qty > 0 
        group by STOR_CD, STOCK_DT , ITEM_CD;
    """
    conn = server_connect(host,port,user,password,db)
    cur = conn.cursor()
    cur.execute(raw_data_query)
    result = cur.fetchall()
    conn.close()
    raw_df = pd.DataFrame(result)

    raw_df['STOCK_DT'] = pd.to_datetime(raw_df['STOCK_DT'], format='%Y%m%d') 

    return raw_df

# 직접 작성하여 사용하다가 이후 사장된 코드
def get_calender_data(host,port,user,password,db):
    query = """
        select DT, YEAR,MONTH,DAY,WEEK_MONTH_STD,DOW ,date_format(str_to_date(dt , '%Y%m%d') , '%a') as DY
        ,case when  substr(month_last_dt, 7,2) = '31' then 1 else 0 end CK_31
        from mst_clndr
        where dt >= '20220101';
    """
    conn = server_connect(host,port,user,password,db)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()

    calender_df = pd.DataFrame(result)
    calender_df = calender_df.loc[calender_df.DT <=(datetime.date.today()+relativedelta(years=1)).strftime("%Y%m%d")]
    calender_df['DT'] = pd.to_datetime(calender_df['DT'],format="%Y%m%d")

    return calender_df

def get_holiday_data(start_year):
    # holiday data read 
    s_year = start_year
    #일반 인증키(Encoding)
    mykey = "jrWjbeUlDWyJVPPe3o94HZqgSSkwQfHI2GThL3Wnjm9tbWgbcpYirrTbhzsxtqGbUU4QJVRqchL4twnGvkg48Q%3D%3D"
    
    list_df = []
    for year in range(s_year,datetime.date.today().year+1):
        for month in range(1,13):
        
            if month < 10:
                month = '0' + str(month)
            else:
                month = str(month)
        
            url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService'
            #공휴일 정보 조회
            operation = 'getRestDeInfo'
            params = {'solYear':year, 'solMonth':month}
    
            request_query = get_request_query(url, operation, params, mykey)
            get_data = requests.get(request_query)    
    
            if True == get_data.ok:
            
                soup = BeautifulSoup(get_data.content, 'html.parser')        
                item = soup.findAll('item')
                for i in item:
                    day = int(i.locdate.string[-2:])
                    weekname = print_whichday(int(year), int(month), day)
                    list_df.append([i.datename.string, i.isholiday.string, i.locdate.string, weekname])
    
    #휴일 정보 데이터프레임화
    holiday = pd.DataFrame(list_df)
    holiday[3] = 1
    holiday = holiday.drop(columns=1)
    holiday.columns = ["holiday","DT","YN"]

    #대체공휴일 전환
    for i in holiday.loc[holiday["holiday"]=="대체공휴일"].index:
        holiday.iloc[i,0] = holiday.iloc[i-1,0]
    #31일 다음날(1일) 혹은 31일인 휴일의경우 YN 값 1로 수정
    holiday = holiday.loc[(holiday.DT <= datetime.date.today().strftime("%Y%m%d")) & (holiday.DT >= '2022-01-01')]
    holiday['DT'] = pd.to_datetime(holiday['DT'], format='%Y%m%d')

    def check_yn(df):
        if (df['DT'] - relativedelta(days=1)).day == 31:
            return 1
        else : 
            return 0 
    holiday['YN'] = holiday.apply(check_yn,axis=1)

    return holiday

# 직접 작성하여 사용하다가 이후 사장된 코드
def raw_data_preprocess(org_df):
    df = org_df.copy()

    df = df.loc[~(df["Deliverdow"]=='0000000')]
    
    df.loc[(df["Deliverdow"]=='0101010')&((df["DOW"]=='3')|(df["DOW"]=='4')),"deliverday"] = 3
    df.loc[(df["Deliverdow"]=='0101010')&((df["DOW"]=='5')|(df["DOW"]=='6')),"deliverday"] = 5
    df.loc[(df["Deliverdow"]=='0101010')&((df["DOW"]=='7')|(df["DOW"]=='1')|(df["DOW"]=='2')),"deliverday"] = 7

    df.loc[(df["Deliverdow"]=='1010100')&((df["DOW"]=='2')|(df["DOW"]=='3')),"deliverday"] = 2
    df.loc[(df["Deliverdow"]=='1010100')&((df["DOW"]=='4')|(df["DOW"]=='5')),"deliverday"] = 4
    df.loc[(df["Deliverdow"]=='1010100')&((df["DOW"]=='6')|(df["DOW"]=='7')|(df["DOW"]=='1')),"deliverday"] = 6

    df.loc[(df["Deliverdow"]=='1101010')&(df['DOW']=='2'),'deliverday'] = 2
    df.loc[(df["Deliverdow"]=='1101010')&((df['DOW']=='3')|(df['DOW']=='4')),'deliverday'] = 3
    df.loc[(df["Deliverdow"]=='1101010')&((df['DOW']=='5')|(df['DOW']=='6')),'deliverday'] = 5
    df.loc[(df["Deliverdow"]=='1101010')&((df['DOW']=='7')|(df['DOW']=='1')),'deliverday'] = 7

    df.loc[(df["Deliverdow"]=='1010110')&((df['DOW']=='2')|(df['DOW']=='3')),'deliverday'] = 2
    df.loc[(df["Deliverdow"]=='1010110')&((df['DOW']=='4')|(df['DOW']=='5')),'deliverday'] = 4
    df.loc[(df["Deliverdow"]=='1010110')&(df['DOW']=='6'),'deliverday'] = 6
    df.loc[(df["Deliverdow"]=='1010110')&((df['DOW']=='7')|(df['DOW']=='1')),'deliverday'] = 7
    
    df["STOR_CD"] = df["STOR_CD"].astype(str)
    df["ITEM_CD"] = df["ITEM_CD"].astype(str)
    df["holiday"] = df["holiday"].astype(str)
    
    df['31day']=df.apply(lambda x : 1 if (x.DAY == 31) else 0,axis=1)
    df = df.loc[df.DT >= "2022-01-01"]
    
    return df

def week_preprocess2(target):
    df = target.copy()
    # 1. 1010100 테이블 전처리 불필요 --> (월화),(수목),(금토일)
    # 2. 1101010 테이블 전처리 --> (월),(화수),(목금),(토일) --> 불필요
    # 3. 1010110 테이블 전처리 --> (월화),(수목),(금),(토일) --> 불필요 
    # 4. 0101010 테이블 전처리 --> (화수)(목금)(토일월) --> 월요일의 week.wms 을 전 주로 맞춰야함
    
    tts_target_idx = df.loc[(df.DELIVERDOW=='0101010')&(df.DOW.isin(['2']))].index.to_list()
    tts_previous_idx = list(map(lambda x : x - 1,tts_target_idx))
    
    for i,(idx,pre_idx) in enumerate(zip(tts_target_idx,tts_previous_idx)):
        #! 첫번째 데이터는 그대로... 임시
        if i == 0 : 
            continue
        df.loc[idx,'WEEK'] = df.loc[pre_idx,'WEEK']
        df.loc[idx,'WEEK_MONTH_STD'] = df.loc[pre_idx,'WEEK_MONTH_STD']
    
    return df

# 팀원의 도움으로 데이터 수집 및 가공 최적화
def get_data_from_db(host, port, user,password, db):
    conn = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=db,
    port=port
    )
    
    with open('./get_data_from_db.sql', 'r') as file:
        sql_script = file.read()
    
    queries = sql_script.split(';')

    result = None
    for query in queries:
        if query.strip() != '':
            with conn.cursor() as cursor:
                cursor.execute(query)
                if query == '\n\n\nselect * from store_stock_5':
                    columns_names = [columns[0] for columns in cursor.description]
                    columns_names = [col.upper() for col in columns_names]
                result = cursor.fetchall()
    
    result_dataframe = pd.DataFrame(columns = columns_names,data = result)
    result_dataframe['DT'] = pd.to_datetime(result_dataframe['DT'],format="%Y%m%d")

    return result_dataframe 

def main():
    parser = argparse.ArgumentParser(description='DATA PRE-PROCESS SCRIPTS')
    parser.add_argument('--date',help='Period of data, Input style is yyyymmdd', default = datetime.date.today().strftime("%Y%m%d"))

    properties = parsers.ConfigParser()
    properties.read("./config.ini")

    host = str(properties['sbp']['host'])
    port = int(properties['sbp']['port'])
    user = str(properties['sbp']['user'])
    password = str(properties['sbp']['password'])
    db = str(properties['sbp']['db'])

    logger.info("GETTERING DATA FROM DATABASE")
    data = get_data_from_db(host=host, port=port, user=user,password=password, db=db)
    logger.info("GETTERING DATA FROM DATABASE DONE")
    holiday = get_holiday_data(start_year=2022)

    final_data = pd.merge(data,holiday,on ="DT",how= "outer")
    final_data = final_data.fillna(0,inplace=False)
    final_data = final_data.sort_values(['STOR_CD','ITEM_CD','DT']).reset_index(drop=True)
    final_data = week_preprocess2(final_data)

    #! 데이터 타입 혼용된 경우 강제로 분리가 발생하여 강제 형변환

    logger.info(f'**** Datasets save to ../datasets ****')
    # 데이터셋 저장 
    if not Path(f"../datasets").exists():
        Path(f"../datasets").mkdir(parents=False,exist_ok=False)

    final_data.to_csv(f"../datasets/{datetime.date.today().strftime('%Y-%m-%d')}_rawdatasets.csv",index=False)
    # logger.info(f'Datasets save done')


def set_logger():
    logging.basicConfig(
        level='NOTSET',
        format=RICH_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logger = logging.getLogger('rich')
    file_handler = logging.FileHandler(LOG_PATH,mode='a',encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)
    return logger

def handler_exception(exc_type,exc_value,exc_traceback):
    logger = logging.getLogger('rich')
    logger.error('Unexpected exception',
                 exc_info = (exc_type,exc_value,exc_traceback))

if __name__ == '__main__':
    RICH_FORMAT = "[%(filename)s:%(lineno)s] >> %(message)s"
    FILE_HANDLER_FORMAT = "[%(asctime)s]\t%(levelname)s\t[%(filename)s:%(lineno)s]\t>> %(message)s"
    # if not Path(f"../log/ML/{datetime.date.today().strftime('%Y%m')}").exists():
    #     Path(f"../log/ML/{datetime.date.today().strftime('%Y%m')}").mkdir(parents=False,exist_ok=False)
    if not Path(f"../log").exists():
        Path(f"../log").mkdir(parents=False,exist_ok=False)
    
    LOG_PATH = f"../log/pre_process.log"
    
    logger = set_logger()
    sys.excepthook = handler_exception
    logger.info(f' **** Data Preprocess Start ****')
    main()
    logger.info(f'**** Data Preprocess End ****')