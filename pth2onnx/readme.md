# 说明
1. 将你通过[R-Centernet](https://github.com/ZeroE04/R-CenterNet)训练出来的模型放到model文件夹，然后在按照以下顺序运行，生成C++调用模型。
2. 注意，修改pth2onnx.py内的模型文件名为你的模型。
3. r_dla_34-trim.onnx就是最终C++调用模型，这里是单类检测，如果你是多类，对应的你要修改dlanet.py内输出hm。
4. 由于onnxruntime暂不支持dcn，所以请使用无dcn版本的模型。

```Bash
python pth2onnx.py
```	
```Bash
pip3 install onnx
pip3 install onnx-simplifier
```	
```Bash
python -m onnxsim model//r_dla_34.onnx model//r_dla_34-sim.onnx
python remove_initializer_from_input.py --input model//r_dla_34-sim.onnx --out model//r_dla_34-trim.onnx
```	

