from src import *
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# 设置全局绘图风格和背景
#sns.set_style("whitegrid")       # 可选: whitegrid, darkgrid, white, dark, ticks
sns.set_context("paper")          # 控制字体、线条粗细等，可选: paper, notebook, talk, poster
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['svg.fonttype'] = 'none'

def main():
    parser = argparse.ArgumentParser(description="三层神经网络 CIFAR-10 图像分类器")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'hyper_search'],
                        help="选择运行模式：train, test 或 hyper_search")
    parser.add_argument('--data_path', type=str, default="data/cifar-10-batches-py",
                        help="CIFAR-10 数据集所在目录")
    parser.add_argument('--num_epochs', type=int, default=20, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=100, help="每个 mini-batch 大小")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="初始学习率")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="学习率衰减因子")
    parser.add_argument('--hidden_size', type=int, default=100, help="隐藏层单元数 (训练和 test 模式下使用)")
    parser.add_argument('--reg', type=float, default=1e-3, help="L2 正则化强度 (训练和 test 模式下使用)")
    parser.add_argument('--model_file', type=str, default='best_model.pkl', help="模型保存/加载文件名")
    # 针对超参数搜索模式的参数
    parser.add_argument('--lr_list', type=str, default="0.001,0.0005", 
                        help="超参数搜索的学习率列表，多个值以逗号分隔")
    parser.add_argument('--hidden_list', type=str, default="100,50",
                        help="超参数搜索的隐藏层单元数列表，多个值以逗号分隔")
    parser.add_argument('--reg_list', type=str, default="0.001,0.0001",
                        help="超参数搜索的正则化强度列表，多个值以逗号分隔")
    args = parser.parse_args()

    # 加载 CIFAR-10 数据集
    (trainX, trainY), (testX, testY) = load_cifar10(args.data_path)

    if args.mode == 'train':
        # 划分验证集，使用最后5000个样本
        num_val = 5000
        X_val = preprocess_data(trainX[-num_val:])
        y_val = trainY[-num_val:]
        X_train = preprocess_data(trainX[:-num_val])
        y_train = trainY[:-num_val]
        X_test_processed = preprocess_data(testX)

        input_size = X_train.shape[1]  # 32*32*3 = 3072
        output_size = 10
        model = ThreeLayerNN(input_size, args.hidden_size, output_size,
                             activation='relu', reg=args.reg)

        # 开始训练
        loss_history, val_acc_history = train(model, X_train, y_train, X_val, y_val,
                                              num_epochs=args.num_epochs, batch_size=args.batch_size,
                                              learning_rate=args.learning_rate, lr_decay=args.lr_decay)
        plot_training(loss_history, val_acc_history)
        save_model(model, filename=args.model_file)

        print("在测试集上评估...")
        test(model, preprocess_data(testX), testY)

    elif args.mode == 'test':
        X_test_processed = preprocess_data(testX)
        input_size = X_test_processed.shape[1]
        output_size = 10
        with open(args.model_file, 'rb') as f:
            params = pickle.load(f)
        hidden_size = params['W1'].shape[1]
        model = ThreeLayerNN(input_size, hidden_size, output_size, activation='relu')
        model.params = params
        test(model, X_test_processed, testY)

    elif args.mode == 'hyper_search':
        # 划分验证集，使用最后5000个样本
        num_val = 5000
        X_val = preprocess_data(trainX[-num_val:])
        y_val = trainY[-num_val:]
        X_train = preprocess_data(trainX[:-num_val])
        y_train = trainY[:-num_val]

        input_size = X_train.shape[1]
        output_size = 10

        # 将命令行字符串参数转换为列表
        lr_list = parse_list_arg(args.lr_list, float)
        hidden_list = parse_list_arg(args.hidden_list, int)
        reg_list = parse_list_arg(args.reg_list, float)

        # 进行超参数搜索，训练周期可适当减少
        results, best_model, best_params_choice = hyperparameter_search(
            X_train, y_train, X_val, y_val, input_size, output_size,
            lr_list, hidden_list, reg_list,
            num_epochs=args.num_epochs, batch_size=args.batch_size,
            lr_decay=args.lr_decay, model_file=args.model_file
        )
        print("\n超参数搜索结果：")
        for key, acc in results.items():
            print(f"lr={key[0]}, hidden_size={key[1]}, reg={key[2]} => Val Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()


