import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', weight_scale=1e-3, reg=0.0):
        """
        初始化模型参数
        :param input_size: 输入向量维度
        :param hidden_size: 隐藏层单元数
        :param output_size: 分类数(例如 CIFAR-10 为10)
        :param activation: 激活函数类型，可选 'relu' 或 'sigmoid'
        :param weight_scale: 权重初始化缩放因子
        :param reg: L2 正则化强度
        """
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_scale
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_scale
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation
        self.reg = reg

    def forward(self, X):
        """前向传播，返回分类得分"""
        self.cache = {}
        z1 = X.dot(self.params['W1']) + self.params['b1']
        self.cache['z1'] = z1
        if self.activation == 'relu':
            a1 = relu(z1)
        elif self.activation == 'sigmoid':
            a1 = sigmoid(z1)
        else:
            raise ValueError("Unsupported activation function: " + self.activation)
        self.cache['a1'] = a1
        scores = a1.dot(self.params['W2']) + self.params['b2']
        return scores

    def compute_loss(self, scores, y):
        """
        计算交叉熵损失，并加入 L2 正则化项
        :param scores: 网络输出得分 (N, C)
        :param y: 真实标签 (N,)
        :return: 损失值（标量）
        """
        # 数值稳定性处理
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = scores.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        # 添加 L2 正则化
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) +
                                  np.sum(self.params['W2'] ** 2))
        # 缓存中间变量供反向传播使用
        self.cache['probs'] = probs
        self.cache['y'] = y
        self.cache['N'] = N
        return loss

    def backward(self, X):
        """
        反向传播，计算并返回各参数梯度字典
        :param X: 输入数据 (N, D)
        :return: grads, 包含 'W1', 'b1', 'W2', 'b2' 的梯度
        """
        grads = {}
        N = self.cache['N']
        probs = self.cache['probs']
        y = self.cache['y']

        # 计算 softmax 层梯度
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # 后向传播至第二层 (隐藏层 -> 输出层)
        grads['W2'] = self.cache['a1'].T.dot(dscores) + self.reg * self.params['W2']
        grads['b2'] = np.sum(dscores, axis=0)

        # 反向传播到隐藏层
        da1 = dscores.dot(self.params['W2'].T)
        z1 = self.cache['z1']
        if self.activation == 'relu':
            dz1 = da1 * relu_grad(z1)
        elif self.activation == 'sigmoid':
            dz1 = da1 * sigmoid_grad(z1)
        else:
            raise ValueError("Unsupported activation function: " + self.activation)

        grads['W1'] = X.T.dot(dz1) + self.reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0)
        return grads

    def predict(self, X):
        """使用模型预测类别"""
        scores = self.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred