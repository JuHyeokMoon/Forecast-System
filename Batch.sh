#! bin/sh

cd /Users/moonju/Documents/darts/code

export predict_date=14
export date=`date +%Y-%m-%d`				# test : '2023-03-26'	
export date='2023-06-23'				# test : '2023-03-26'	
export raw_data=${date}_rawdatasets.csv 	# test.csv
export raw_data=2023-05-09_rawdatasets.csv
export del_periods=28
export val_date=`date -d '1 week ago' "+%Y-%m-%d"`
export val_date='2023-03-01'


export Pre_Process="bash darts_pre_process.sh"
export Main="bash darts_main.sh .. $raw_data $predict_date 2022-07-01 $date"
export Post_Process="bash darts_post_process.sh .. $raw_data ${date}_pred_raw.csv $date"
export Verification="bash darts_verification.sh .. $raw_data ${val_date}_pred_raw.csv $val_date"
export Filemanagement="bash file_management.sh $del_periods"


echo "Process Start Time : $(date)" >> ../log/batch_monitoring.log

{
	$Pre_Process &&
	echo "Pre_Process Done." >> ../log/batch_monitoring.log
} || {
    echo "[ ERROR ] Please check Pre_Process." >> ../log/batch_monitoring.log
}
{
	$Main &&
	echo "Training & Forecasting Done." >> ../log/batch_monitoring.log
} || {
    echo "[ ERROR ] Please check Main." >> ../log/batch_monitoring.log
}
{
	$Post_Process &&
	echo "Post Processing Done." >> ../log/batch_monitoring.log
} || {
    echo "[ ERROR ] Please check Post_Process." >> ../log/batch_monitoring.log
}
{
	$Verification &&
	echo "Verification Done." >> ../log/batch_monitoring.log
} || {
    echo "[ ERROR ] Please check Verification." >> ../log/batch_monitoring.log
}
{
	$Filemanagement &&
	echo "Filemanagement Done." >> ../log/batch_monitoring.log
} || {
    echo "[ ERROR ] Please check Filemanagement." >> ../log/batch_monitoring.log
}

echo "Process End Time : $(date)" >> ../log/batch_monitoring.log
echo '-----------------------------' >> ../log/batch_monitoring.log
