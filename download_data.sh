export DATA=./datasets/iclr2022
mkdir -p $DATA
aws s3 cp s3://zav-public/multi-obj-repr-learning/iclr2022/ $DATA --recursive

export DATA=./datasets/training-data
mkdir -p $DATA
aws s3 cp s3://zav-public/multi-obj-repr-learning/training-data/ $DATA --recursive