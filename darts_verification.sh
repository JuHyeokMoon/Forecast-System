#! /bin/sh
# BR Smart Store 검증 스크립트
# bash darts_verification.sh .. 2023-02-01 2023-02-01_nlinear_14.csv 20230414_rawdatasets.csv

if [[ $1 && $2 && $3 && $4 ]]

then
    path=$1                      # Input 데이터로 구성할 시작 날짜
    actual_data=$2               # 실제값 데이터 (출고 타입 사용)
    pred_data=$3                 # 예측치 데이터 (출고량 데이터)
    pred_start_date=$4           # Input 데이터로 구성할 끝 날짜
else
    echo "[Error] Check Input Parameter. (ex darts_verification.sh [path] [pred_start_date] [pred_data] [actual_data])"
    exit
fi

python darts_verification.py \
    --path $path \
    --pred_start_date $pred_start_date \
    --pred_data $pred_data \
    --actual_data $actual_data \

