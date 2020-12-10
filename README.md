# License Plate Detection with RetinaFace

距离上次车牌检测模型更新已经过了一年多的时间，这段时间也有很多快速、准确的模型提出，我们利用单物体检测算法Retinaface进行了车牌检测模型的训练，通过测试，检测效果和适用性都更突出，支持的模型也更为丰富。

我们开源版本的检测算法经过了多个版本迭代，考虑到检测的效率跟准确率，原始版本逐步淘汰，从最初的基于LBP和Harr特征的车牌检测，感兴趣的小伙伴可以参考 train-detector(https://github.com/openalpr/train-detector) 这个仓库；到后来逐步的采用深度学习的方式，我们的上一个版本采用基于mobilenet-ssd的算法进行检测，大家可以移步 (https://gitee.com/zeusees/Mobilenet-SSD-License-Plate-Detection) 这里进行查看，后续请尽量采用新模型进行测试。

该版本的检测模型的训练，结合了CCPD数据集跟我们自有的数据，能够做到更多车牌种类的支持。

### Pytorch模型测试
##### Clone and install
1. git clone https://github.com/zeusees/License-Plate-Detector.git

2. Pytorch version 1.2.0

3. Python 3.6

4. python detect.py


### 基于C++的NCNN模型测试
##### Source Code Compile
1. cd Prj-ncnn

2. cmake .

3. make


### 支持车牌种类

- 蓝色单层车牌
- 黄色单层车牌
- 绿色新能源车牌、民航车牌
- 黑色单层车牌
- 白色警牌、军牌、武警车牌
- 黄色双层车牌
- 绿色农用车牌
- 白色双层军牌


### 测试结果

![](imgs/res.jpg)


### 参考
- [Retinaface (Pytorch)](https://github.com/biubug6/Pytorch_Retinaface)
- [Pytorch_Retina_License_Plate](https://github.com/gm19900510/Pytorch_Retina_License_Plate)
- [CCPD](https://github.com/detectRecog/CCPD)

