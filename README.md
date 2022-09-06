百度网盘AI大赛-模糊文档图像恢复赛第13名方案
使用移动设备扫描获取文档的过程中，很多是文字、字母和数字等符号内容。通过拍摄截取等方式获取文档，就非常有可能导致内容模糊、噪音叠加等问题的发生，使得无法实际发挥作用。期望同学们通过计算机技术，帮助人们将模糊失焦的文档恢复清晰，提高使用便捷度和效率。
目的是帮助人们将模糊失焦的文档恢复清晰，提高使用便捷度和效率。

项目描述
采用MIRNetV2进行训练，然后将预测图片resize到(2048,1024)，进行8组预测，最后再将预测的结果拼接在一起，提升了很多性能

项目结构
├── main_MIRNetV2.ipynb
├── README.md
├── predict.py
使用方式
A：在AI Studio上运行本项目
B：训练：修改构造数据读取器的路径，在main_MIRNetV2.ipynb进行修改。如：模糊图片的路径data/data154549/train_data_01/train_data_01/blur_image/
真实图片的路径：data/data154549/train_data_01/train_data_01/gt_image/
代码如下:
 def __init__(self, mode = 'train', watermark_dir = 'data/data154549/train_data_01/train_data_01/blur_image/', bg_dir = 'data/data154549/train_data_01/train_data_01/gt_image/'):
C:预测
命令行运行测代码，主要需要在predict.py修改使用模型的路径，如：param_dict = paddle.load('./MIRnetV2_model_5.pdparams')
python predict.py data/data154549/train_data_01/deblur_testA/blur_image results"# Blurred_Document_Image_Recovery_MIRNetV2" 
# Blurred_Document_Image_Recovery_MIRNetV2
