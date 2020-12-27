# R-CenterNet_onnxruntime
R-CenterNet's inference.(C++)


# 说明
1. 确保cuda10、cudnn7、opencv已经安装了
```Bash
#查看cuda版本
cat /usr/local/cuda/version.txt
#查看cudnn版本
nvcc -V
#查看opencv版本
pkg-config --modversion opencv
```	
如果没装，自己百度安装好上面三个。

2. 安装onnxruntime
``` Bash
sh ./onnx_install/onnxruntime_install.sh
```
onnxruntime安装完毕请将./local/lib/目录下的**libonnxruntime.so**拷贝到此目录下

3. 编译predict.cpp
``` Bash
mkdir build && cd build
cmake ..
make install predict
```

4. 开始推理
``` Bash
./predict your_model_path//your_model.onnx your_img_path//your_img.jpg
```
[注]
1. onnxruntime安装完毕请将./local/lib/目录下的libonnxruntime.so拷贝到此目录下
2. 本样例资源在[**some_resource**](https://github.com/ZeroE04/some_resource)处下载
