## conda虚拟环境 -- smart_cockpit_env
代码需要运行在 python 版本为 3.8的 conda 虚拟环境中
- 使用以下命令来创建虚拟环境
  
  `conda create -n smart_cockpit_env python=3.8`
- 使用以下命令来安装代码所需要的库
  - opencv 库的安装

    ` pip install opencv-python==4.12.0.88`
  - dlib 库的安装

    `pip install dlib==20.0.0`
  
    注意，这里安装的是20.0.0版本，前面的19.x.x版本尝试安装，但是安装失败了，可能是兼容性的问题，后面再来尝试解决，反正20.0.0可以正常安装，没出问题的话就用20.0.0版本吧。
  - pandas 库的安装
  
    `pip install pandas==1.4.4`
  -  pyserial 库的安装

    `pip install pyserial==3.5`
  - tqdm 库的安装
  
    `pip install tqdm==4.67.1` 

## 模型资源文件 -- model_resources
- 关于模型所需要的资源文件（共3个）,详见 model_resources文件夹，该文件夹可通过 [百度网盘链接](https://pan.baidu.com/s/1BwtISWhscuxLT7Kx7lqIag?pwd=xxxt)获取。
