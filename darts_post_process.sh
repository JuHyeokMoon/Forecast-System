#! /bin/sh
# Smart Store 후처리 스크립트
# 학습한 데이터(actual_data)로 case 2 에 해당하는 후처리 진행하여 예측치 가공
# bash darts_post_process.sh .. 2023-02-01 2023-02-01_nlinear_14.csv 20230414_rawdatasets.csv

if [[ $1 && $2 && $3 && $4 ]]

then
    path=$1                      # Input 데이터로 구성할 시작 날짜
    actual_data=$2               # 실제값 데이터 (출고 타입 사용)
    pred_data=$3                 # 예측치 데이터 (출고량 데이터)
    pred_start_date=$4           # Input 데이터로 구성할 끝 날짜
else
    echo "[Error] Check Input Parameter. (ex darts_post_process.sh [path] [pred_start_date] [pred_data] [actual_data])"
    exit
fi

python darts_post_process.py \
    --path $path \
    --pred_start_date $pred_start_date \
    --pred_data $pred_data \
    --actual_data $actual_data \