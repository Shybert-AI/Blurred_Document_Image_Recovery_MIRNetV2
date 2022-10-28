# 百度网盘AI大赛-模糊文档图像恢复赛第13名方案

MIRNetV2: Learning Enriched Features for Fast Image Restoration and Enhancement  

官方源码： [https://github.com/swz30/MIRNetV2](https://github.com/swz30/MIRNetV2)

复现地址：[https://github.com/sldyns/MIRNetV2_paddle](https://github.com/sldyns/MIRNetV2_paddle)

## 1. 项目描述

采用MIRNetV2进行训练，然后将预测图片resize到(2048,1024)，进行8组预测，最后再将预测的结果拼接在一起，提升了很多性能

## 2. 精度

## score	    psnr	        ms_ssim		
## 0.57	      20.32433	    0.93676	

## 3. 文件结构

### 文件结构

```
├── main_MIRNetV2.ipynb   #训练代码
├── README.md
├── predict.py            #推理代码
```

### 模型预测

```shell
# MIRNet V2
python predict.py data/data154549/train_data_01/deblur_testA/blur_image results
```


