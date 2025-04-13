import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .data import *
from .model import *
def train(model, X_train, y_train, X_val, y_val,
          num_epochs=20, batch_size=100, learning_rate=1e-3, lr_decay=0.95):
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    best_val_acc = 0
    best_params = {}
    loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        X_train, y_train = shuffle_data(X_train, y_train)
        for i in range(iterations_per_epoch):
            start = i * batch_size
            end = (i + 1) * batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 前向传播并计算损失
            scores = model.forward(X_batch)
            loss = model.compute_loss(scores, y_batch)
            loss_history.append(loss)

            # 反向传播及参数更新（SGD）
            grads = model.backward(X_batch)
            for key in model.params:
                model.params[key] -= learning_rate * grads[key]

        learning_rate *= lr_decay
        y_val_pred = model.predict(X_val)
        val_acc = np.mean(y_val_pred == y_val)
        val_acc_history.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}, Val Accuracy = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

    model.params = best_params
    return loss_history, val_acc_history


def plot_training(loss_history, val_acc_history):
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./asset/loss.jpg",dpi=300)
    plt.show()

    plt.figure()
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./asset/acc.jpg",dpi=300)
    plt.show()


def save_model(model, filename='best_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model.params, f)
    print("模型参数已保存至", filename)


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print("测试集准确率: {:.4f}".format(acc))


def hyperparameter_search(X_train, y_train, X_val, y_val, input_size, output_size, 
                          lr_list, hidden_list, reg_list, num_epochs, batch_size, lr_decay, model_file):
    results = {}
    best_val = -1
    best_model = None
    best_params_choice = None

    for lr in lr_list:
        for hs in hidden_list:
            for reg in reg_list:
                print(f"训练中：lr={lr}, hidden_size={hs}, reg={reg}")
                model = ThreeLayerNN(input_size, hs, output_size, activation='relu', reg=reg)
                _, val_acc_history = train(model, X_train, y_train, X_val, y_val,
                                           num_epochs=num_epochs, batch_size=batch_size,
                                           learning_rate=lr, lr_decay=lr_decay)
                curr_val_acc = val_acc_history[-1]
                results[(lr, hs, reg)] = curr_val_acc
                print(f"验证集准确率: {curr_val_acc:.4f}")
                if curr_val_acc > best_val:
                    best_val = curr_val_acc
                    best_model = model
                    best_params_choice = (lr, hs, reg)
    
    print(f"\n最佳验证准确率: {best_val:.4f}，参数组合: lr={best_params_choice[0]}, hidden_size={best_params_choice[1]}, reg={best_params_choice[2]}")
    save_model(best_model, filename=model_file)
    return results, best_model, best_params_choice


def parse_list_arg(arg_str, type_func=float):
    """
    将形如 "0.001,0.0005" 的字符串解析为列表 [0.001, 0.0005]
    默认转换为 float，可通过 type_func 指定类型，如 int
    """
    return [type_func(x) for x in arg_str.split(",")]