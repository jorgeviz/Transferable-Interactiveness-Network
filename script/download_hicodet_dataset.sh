#!/bin/bash

# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# ---------------HICO-DET Dataset------------------
echo "Downloading HICO-DET Dataset"

python scriptDownload_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

echo "Downloading HICO-DET Evaluation Code"
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp script/Generate_detection.m Data/ho-rcnn/
cp script/save_mat.m Data/ho-rcnn/
cp script/load_mat.m Data/ho-rcnn/

mkdir Data/ho-rcnn/data/hico_20160224_det/
cp Data/hico_20160224_det/anno_bbox.mat Data/ho-rcnn/data/hico_20160224_det/
cp Data/hico_20160224_det/anno.mat Data/ho-rcnn/data/hico_20160224_det/

mkdir -Results/

echo "Downloading training data..."
python script/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
python script/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
python script/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
python script/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
python script/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
python script/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
python script/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
python script/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
python script/Download_data.py 12-ZFl2_AwRkVpRe5sqOJRJrGzBeXtWLm Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl
python script/Download_data.py 1y3cnbX12jwNAoSiXDLcdzn-nF_jvsaum Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl
python script/Download_data.py 1IGxFW2Fe8uFRSoDtg5k_7dOxR7lK6SIA Data/Trainval_GT_HICO_with_pose.pkl
python script/Download_data.py 1o59JGvhzI7CzdIVSxbcq0L6UnMLX0tDO Data/Trainval_GT_VCOCO_with_pose.pkl
python script/Download_data.py 1qioujFz-jWBXPb-gCRqh6ttpf79H06Re Data/Trainval_Neg_HICO_with_pose.pkl
python script/Download_data.py 1zEJqQcq2KF5QC8D2TIFAo0zog0PH2LF- Data/Trainval_Neg_VCOCO_with_pose.pkl
python script/Download_data.py 1sjV6e916NIPcYYqbGwhKM6Vhl7SY6WqD -Results/80000_TIN_D_noS.pkl
python script/Download_data.py 1sJipmoZ-5u0ymm8diqYd5Yqk2A-QQBXN -Results/60000_TIN_VCOCO_D.pkl
