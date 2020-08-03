#### 安装要求

python3 \ pytorch


#### 准备内容

将待预测文件解压至./testData/目录，并以文件夹形式存放


#### 运行方式

python run.py


#### 参数介绍

参见 ./config.py

其中：

cfg.device : 表示选用的GPU ID，可任意替换

cfg.saveFolder : 表示保存路径，默认为'./predict/'，可任意替换

cfg.checkpoint : 表示模型路径，默认为'./MetNet/model_best.pth.tar'，无需修改

cfg.filePath : 表示待预测样本名称，默认为"./MetNet/test.txt"，无需修改

cfg.dataFolder : 表示待预测样本路径，默认为'./testData/'，可将'./testData/'替换为任意路径



#### 注：

初赛单模型评分为0.2473，采用5个epoch模型预测结果集成后，评分为排行榜的0.2494

模型实现部分代码fork自： https://github.com/Hzzone/Precipitation-Nowcasting
