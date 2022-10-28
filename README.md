# 百度网盘AI大赛-模糊文档图像恢复赛第13名方案

# 大赛背景  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;百度网盘AI大赛——图像处理挑战赛是百度网盘开放平台面向开发者发起的图像处理挑战赛事，旨在基于个人云存储的生态能力开放，通过比赛机制，鼓励选手结合当下的前沿图像处理与计算机视觉技术，设计合适模型，并提升模型的效果，助力互联网用户更便捷地进行数字生活、学习和工作，为中国开源生态建设贡献力量。本次图像处理挑战赛以线上比赛的形式进行，参赛选手在规定时间内提交基于评测数据集产出的结果文件，榜单排名靠前并通过代码复查的队伍可获得高额奖金。百度网盘开放平台致力于为全球AI开发者和爱好者提供专业、高效的AI学习和开发环境，挖掘培养AI人才，助力技术产业生态发展。  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此次大赛主题结合日常生活常见情景展开，当使用移动设备扫描获取文档的过程中，很多是文字、字母和数字等符号内容。通过拍摄截取等方式获取文档，就非常有可能导致内容模糊、噪音叠加等问题的发生，使得无法实际发挥作用。期望同学们通过计算机技术，帮助人们将模糊失焦的文档恢复清晰，提高使用便捷度和效率。  

MIRNetV2: Learning Enriched Features for Fast Image Restoration and Enhancement  

官方源码： [https://github.com/swz30/MIRNetV2](https://github.com/swz30/MIRNetV2)

复现地址：[https://github.com/sldyns/MIRNetV2_paddle](https://github.com/sldyns/MIRNetV2_paddle)

## 1. 项目描述

采用MIRNetV2进行训练，然后将预测图片resize到(2048,1024)，进行8组预测，最后再将预测的结果拼接在一起，提升了很多性能

## 2. 模型的精度

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


