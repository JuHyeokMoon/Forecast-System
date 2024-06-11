#! /bin/sh
# BR Smart Store 2차 POC 메인 스크립트
# bash darts_main.sh .. 20230414_rawdatasets.csv 14 2022-07-01 2023-02-01

if [[ $1 && $2 && $3 && $4 && $5 ]]

then
    path=$1                 # Input 데이터로 구성할 시작 날짜
    raw_data=$2             # Input 데이터로 구성할 끝 날짜
    pred_date=$3            # 출고량 데이터
    train_start_date=$4     # 출고 타입 데이터
    pred_start_date=$5      # 출고 타입 데이터
else
    echo "[Error] Check Input Parameter. (ex darts_main.sh [path] [raw_data] [pred_date] [train_start_date] [pred_start_date])"
    exit
fi

python darts_main.py \
    --path $path \
    --raw_data $raw_data \
    --pred_date $pred_date \
    --train_start_date $train_start_date \
    --pred_start_date $pred_start_date
