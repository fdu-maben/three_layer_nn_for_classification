import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 设置全局绘图风格和背景
#sns.set_style("whitegrid")       # 可选: whitegrid, darkgrid, white, dark, ticks
sns.set_context("paper")          # 控制字体、线条粗细等，可选: paper, notebook, talk, poster
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['svg.fonttype'] = 'none'

# 加载模型权重
with open('./asset/model.pkl', 'rb') as f:
    model_params = pickle.load(f)

# 提取参数
W1 = model_params['W1']   # 第一层权重，形状为 (3072, hidden_size)
W2 = model_params['W2']   # 第二层权重，形状为 (hidden_size, output_size)

# -------------------------------
# 可视化 1：直方图展示第一层权重的分布
# -------------------------------
plt.figure()
plt.hist(W1.flatten(), bins=50)
plt.title('Histogram of W1 Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("./asset/distribution_of_layer1_weight.jpg",dpi=300)
plt.show()

# -------------------------------
# 可视化 2：将第一层权重中的部分列还原为图像，观察滤波器样式
# -------------------------------
# 因为输入图像尺寸为 32x32x3，故每个隐藏神经元对应一个长向量，我们可以将其reshape为 (32, 32, 3)
hidden_size = W1.shape[1]
num_filters = min(10, hidden_size)  # 显示前 10 个滤波器

plt.figure(figsize=(15, 6))
for i in range(num_filters):
    weight_vector = W1[:, i]
    # 将该向量还原为 (32, 32, 3) 的图像
    filter_image = weight_vector.reshape(32, 32, 3)
    
    # 对图像进行归一化处理（0-1），便于可视化
    filter_min = filter_image.min()
    filter_max = filter_image.max()
    normalized_filter = (filter_image - filter_min) / (filter_max - filter_min + 1e-8)
    
    plt.subplot(2, (num_filters + 1) // 2, i + 1)
    plt.imshow(normalized_filter)
    plt.title(f'Filter {i + 1}')
    plt.axis('off')
plt.suptitle('Visualization of First Layer Filters')
plt.tight_layout()
plt.savefig("./asset/Visualization of First Layer Filters.jpg",dpi=300)
plt.show()

# -------------------------------
# 第二层权重的分布
# -------------------------------
plt.figure()
plt.hist(W2.flatten(), bins=50)
plt.title('Histogram of W2 Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("./asset/layer2_weighr.jpg",dpi=300)
plt.show()
