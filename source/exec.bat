# initialize
echo "Starting script - train A vs each category"
mkdir /home/itamargoz/data/tmp/
mkdir /home/itamargoz/data/tmp/train
mkdir /home/itamargoz/data/tmp/test
mkdir /home/itamargoz/data/tmp/validation

mv /home/itamargoz/data/sorted/seg/train/treat_b /home/itamargoz/data/tmp/train/treat_b
mv /home/itamargoz/data/sorted/seg/train/treat_c /home/itamargoz/data/tmp/train/treat_c
mv /home/itamargoz/data/sorted/seg/train/treat_d /home/itamargoz/data/tmp/train/treat_d

mv /home/itamargoz/data/sorted/seg/test/treat_b /home/itamargoz/data/tmp/test/treat_b
mv /home/itamargoz/data/sorted/seg/test/treat_c /home/itamargoz/data/tmp/test/treat_c
mv /home/itamargoz/data/sorted/seg/test/treat_d /home/itamargoz/data/tmp/test/treat_d

mv /home/itamargoz/data/sorted/seg/validation/treat_b /home/itamargoz/data/tmp/validation/treat_b
mv /home/itamargoz/data/sorted/seg/validation/treat_c /home/itamargoz/data/tmp/validation/treat_c
mv /home/itamargoz/data/sorted/seg/validation/treat_d /home/itamargoz/data/tmp/validation/treat_d

echo "mkdir and mv is done! training start"
# start training
echo "#### A vs B"
## prepare data
mv /home/itamargoz/data/tmp/train/treat_b /home/itamargoz/data/sorted/seg/train/treat_b
mv /home/itamargoz/data/tmp/test/treat_b /home/itamargoz/data/sorted/seg/test/treat_b
mv /home/itamargoz/data/tmp/validation/treat_b /home/itamargoz/data/sorted/seg/validation/treat_b
## making sure only desired folder exists
echo "train: " `ls /home/itamargoz/data/sorted/seg/train/`
echo "test: " `ls /home/itamargoz/data/sorted/seg/test/`
echo "validation: " `ls /home/itamargoz/data/sorted/seg/validation/`
## train
# python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py CIFAR10_AB 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ab.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ab_err.log
python /home/itamargoz/trunk/Banana-Learning/source/utils/predict_analysis.py CIFAR10_AB A,B 1> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ab.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ab_err.log
## restore
mv /home/itamargoz/data/sorted/seg/train/treat_b /home/itamargoz/data/tmp/train/treat_b
mv /home/itamargoz/data/sorted/seg/test/treat_b /home/itamargoz/data/tmp/test/treat_b
mv /home/itamargoz/data/sorted/seg/validation/treat_b /home/itamargoz/data/tmp/validation/treat_b

echo "#### A vs C"
## prepare data
mv /home/itamargoz/data/tmp/train/treat_c /home/itamargoz/data/sorted/seg/train/treat_c
mv /home/itamargoz/data/tmp/test/treat_c /home/itamargoz/data/sorted/seg/test/treat_c
mv /home/itamargoz/data/tmp/validation/treat_c /home/itamargoz/data/sorted/seg/validation/treat_c
## making sure only desired folder exists
echo "train: " `ls /home/itamargoz/data/sorted/seg/train/`
echo "test: " `ls /home/itamargoz/data/sorted/seg/test/`
echo "validation: " `ls /home/itamargoz/data/sorted/seg/validation/`
## train
# python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py CIFAR10_AC 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ac.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ac_err.log
python /home/itamargoz/trunk/Banana-Learning/source/utils/predict_analysis.py CIFAR10_AC A,C 1> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ac.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ac_err.log
## restore
mv /home/itamargoz/data/sorted/seg/train/treat_c /home/itamargoz/data/tmp/train/treat_c
mv /home/itamargoz/data/sorted/seg/test/treat_c /home/itamargoz/data/tmp/test/treat_c
mv /home/itamargoz/data/sorted/seg/validation/treat_c /home/itamargoz/data/tmp/validation/treat_c

echo "#### A vs D"
## prepare data
mv /home/itamargoz/data/tmp/train/treat_d /home/itamargoz/data/sorted/seg/train/treat_d
mv /home/itamargoz/data/tmp/test/treat_d /home/itamargoz/data/sorted/seg/test/treat_d
mv /home/itamargoz/data/tmp/validation/treat_d /home/itamargoz/data/sorted/seg/validation/treat_d
## making sure only desired folder exists
echo "train: " `ls /home/itamargoz/data/sorted/seg/train/`
echo "test: " `ls /home/itamargoz/data/sorted/seg/test/`
echo "validation: " `ls /home/itamargoz/data/sorted/seg/validation/`
## train
# python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py CIFAR10_AD A,D 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ad.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ad_err.log
python /home/itamargoz/trunk/Banana-Learning/source/utils/predict_analysis.py CIFAR10_AD 1> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ad.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/predict_analysis_ad_err.log
## restore
mv /home/itamargoz/data/sorted/seg/train/treat_d /home/itamargoz/data/tmp/train/treat_d
mv /home/itamargoz/data/sorted/seg/test/treat_d /home/itamargoz/data/tmp/test/treat_d
mv /home/itamargoz/data/sorted/seg/validation/treat_d /home/itamargoz/data/tmp/validation/treat_d

echo "finalizing steps began"
# resore all
mv /home/itamargoz/data/tmp/train/treat_b /home/itamargoz/data/sorted/seg/train/treat_b
mv /home/itamargoz/data/tmp/train/treat_c /home/itamargoz/data/sorted/seg/train/treat_c
mv /home/itamargoz/data/tmp/train/treat_d /home/itamargoz/data/sorted/seg/train/treat_d

mv /home/itamargoz/data/tmp/test/treat_b /home/itamargoz/data/sorted/seg/test/treat_b
mv /home/itamargoz/data/tmp/test/treat_c /home/itamargoz/data/sorted/seg/test/treat_c
mv /home/itamargoz/data/tmp/test/treat_d /home/itamargoz/data/sorted/seg/test/treat_d

mv /home/itamargoz/data/tmp/validation/treat_b /home/itamargoz/data/sorted/seg/validation/treat_b
mv /home/itamargoz/data/tmp/validation/treat_c /home/itamargoz/data/sorted/seg/validation/treat_c 
mv /home/itamargoz/data/tmp/validation/treat_d /home/itamargoz/data/sorted/seg/validation/treat_d

rm -r /home/itamargoz/data/tmp/train
rm -r /home/itamargoz/data/tmp/test
rm -r /home/itamargoz/data/tmp/validation
rm -r /home/itamargoz/data/tmp/

echo "all done!"
