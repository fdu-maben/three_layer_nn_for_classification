# 三层神经网络分类器

该项目实现了一个从零开始构建的三层神经网络分类器，旨在使用 CIFAR-10 数据集进行图像分类。项目包括数据准备、模型训练、超参数搜索、测试以及可视化分析等步骤。

## 目录结构

```
.
├── asset/            # 存放图表、日志文件及模型权重
├── data/             # 存放 CIFAR-10 数据集
├── src/              # 源代码目录
├── parameter_search.sh # 用于超参数搜索的脚本
├── train.sh          # 用于训练模型的脚本
├── test.sh           # 用于测试模型的脚本
└── visualize_nn.py    # 用于可视化模型结果的脚本
```

## 数据准备

### 步骤 1：克隆仓库

首先克隆本项目仓库到本地：

```bash
git clone https://github.com/fdu-maben/three_layer_nn_for_classification.git
```

### 步骤 2：下载模型权重

下载已训练的模型权重文件，并将其保存到项目中的 `asset` 文件夹下：

[下载链接](https://drive.google.com/file/d/1hUuSn3FNFwNqq42Icb-uz7ifpDhGjZRL/view?usp=drive_link)

### 步骤 3：准备 CIFAR-10 数据集

CIFAR-10 数据集将被存储在 `data` 文件夹中。如果数据集尚未下载，可以通过以下方式获取：

```bash
# 下载 CIFAR-10 数据集并解压到 data 目录
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
# 解压数据集
tar -xvzf data/cifar-10-python.tar.gz -C data/
# 删除压缩文件以节省空间
rm data/cifar-10-python.tar.gz
```

## 超参数搜索

运行 `parameter_search.sh` 脚本进行超参数搜索，以优化模型性能。

```bash
sh parameter_search.sh
```

此步骤将自动测试不同的超参数组合，并记录结果。

## 模型训练

使用以下命令开始训练三层神经网络模型：

```bash
sh train.sh
```

此脚本将执行模型训练过程，并在 `asset` 文件夹中保存训练过程中的日志和模型权重。

## 模型测试

在训练完成后，使用 `test.sh` 脚本进行模型测试，并评估模型在 CIFAR-10 测试集上的表现：

```bash
sh test.sh
```

该脚本将输出模型的准确率，并生成相应的测试报告。

## 模型分析

使用以下命令生成模型分析结果并可视化训练过程中的关键指标：

```bash
python visualize_nn.py
```

此脚本将生成训练过程中各类图表，如损失函数曲线、准确率变化等，帮助分析模型的训练效果。
