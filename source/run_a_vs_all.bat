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
mv /home/itamargoz/data/tmp/validation/treat_b/home/itamargoz/data/sorted/seg/validation/treat_b
## train
python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ab.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ab_err.log 
## restore
mv /home/itamargoz/data/sorted/seg/train/treat_b /home/itamargoz/data/tmp/train/treat_b
mv /home/itamargoz/data/sorted/seg/test/treat_b /home/itamargoz/data/tmp/test/treat_b
mv /home/itamargoz/data/sorted/seg/validation/treat_b /home/itamargoz/data/tmp/validation/treat_b

echo "#### A vs C"
## prepare data
mv /home/itamargoz/data/tmp/train/treat_c /home/itamargoz/data/sorted/seg/train/treat_c
mv /home/itamargoz/data/tmp/test/treat_c /home/itamargoz/data/sorted/seg/test/treat_c
mv /home/itamargoz/data/tmp/validation/treat_c/home/itamargoz/data/sorted/seg/validation/treat_c
## train
python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ac.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ac_err.log 
## restore
mv /home/itamargoz/data/sorted/seg/train/treat_c /home/itamargoz/data/tmp/train/treat_c
mv /home/itamargoz/data/sorted/seg/test/treat_c /home/itamargoz/data/tmp/test/treat_c
mv /home/itamargoz/data/sorted/seg/validation/treat_c /home/itamargoz/data/tmp/validation/treat_c

echo "#### A vs D"
## prepare data
mv /home/itamargoz/data/tmp/train/treat_d /home/itamargoz/data/sorted/seg/train/treat_d
mv /home/itamargoz/data/tmp/test/treat_d /home/itamargoz/data/sorted/seg/test/treat_d
mv /home/itamargoz/data/tmp/validation/treat_d/home/itamargoz/data/sorted/seg/validation/treat_d
## train
python /home/itamargoz/trunk/Banana-Learning/source/native_cnn_2ct.py 1> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ad.log 2> /home/itamargoz/trunk/Banana-Learning/source/logs/native_main_run_ad_err.log 
## restore

echo "finalizing steps began"
mv /home/itamargoz/data/sorted/seg/train/treat_d /home/itamargoz/data/tmp/train/treat_d
mv /home/itamargoz/data/sorted/seg/test/treat_d /home/itamargoz/data/tmp/test/treat_d
mv /home/itamargoz/data/sorted/seg/validation/treat_d /home/itamargoz/data/tmp/validation/treat_d

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

rm /home/itamargoz/data/tmp/train
rm /home/itamargoz/data/tmp/test
rm /home/itamargoz/data/tmp/validation
rm /home/itamargoz/data/tmp/

echo "all done!"