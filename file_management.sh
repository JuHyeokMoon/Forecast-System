#! bin/sh
# 파일이 생성된 지 일정기간이 지난 파일은 삭제. 
# periods + 1 일 이전(+) 파일들을 삭제한다.

if [[ $1 ]]
then 
    periods=$1
else
    echo "[Error] Check Input Parameter. (ex file_management.sh 27 )"
    exit
fi

echo "[START delete previous files]"
# 데이터 셋
find ../datasets/ -mtime +$periods -name '*.csv' -exec rm -rf {} \;

echo "[END delete previous files]"