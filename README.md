# 1.开源库版本
 ubuntu 20.04
 
 gcc 9.4.0
 
 cmake 3.16.3

 boost 1.62.0
 
 opencv 3.4.20
 
 Eigen 3.2.7
 
 ceres 1.14.0
 
 glog 0.7.1
 
 spii master维护版本
# 2.配置路径
 修改：path/main/main.cpp的64行的kDir以及94行更改为视频文件的名字
 
 文件部署：
 
 1）视频文件放置在kDir配置的路径下 
 
 2）在可执行程序的上层build的文件夹路径下配置doc文件夹，并放置cameraintrinsics_eye.txt
# 3.瞳孔识别添加了pupil的修改记录
 path/main/main.cpp下修改254-263行
 
 瞳孔识别中丢失估计视轴的问题预计出现在263行的pupilFitter.pupilAreaFitRR方法中，这里将其暴力修改为262的强行置true

