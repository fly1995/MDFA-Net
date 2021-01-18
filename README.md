# MDFA_Net cardiac segmentation
This paper "MDFA-Net: Multiscale dual-path feature aggregation network for cardiac segmentation on Multi-sequence Cardiac MR" has been accepted by Knowledge-Based Systems. If you want to use this code, please cite this paper.https://www.sciencedirect.com/science/article/pii/S0950705121000393

# Data 
The data are provided by the organizer of the Multi-sequence Cardiac MR Segmentation Challenge (MS-CMRSeg 2019) in conjunction with 2019 Medical Image Computing and Computer Assisted Interventions (MICCAI). We also conducted external validation experimnets on the data of 2020 MICCAI myocardial pathology segmentation challenge (MyoPS 2020).  You can download data from http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg19/ and http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/data1.html

# Requirements
python	3.7.9	
keras	2.2.4	
tensorflow-gpu	1.13.1
jieba	0.42.1	
matplotlib	3.3.2	
nibabel	3.2.0	
numpy	1.19.2	
opencv-python	4.4.0.44	
pandas	1.2.0	
pillow	8.0.1	
scikit-image	0.17.2	
scikit-learn	0.23.2	
scipy	1.5.3	
simpleitk	2.0.1	

# To run the experiments
Data_format_conversion.py  You can transform jpg to npy, nii to npy, npy tp jpg, npy tp mat.
Data_preprocess.py   You can propocess the data step by step according to this file. Including augmentation.
Network.py   All network are compared in this paper.
Train_network.py   Train network(after Data_preprocess.py).
predict time.py   Compute prediction time.
Predict.py   Predict model (get test result).
Metrics.py   All metrics are used in this paper.




