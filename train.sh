nohup python -u main.py \
    --mode train \
    --data_path ./data/cifar-10-batches-py \
    --num_epochs 200 \
    --batch_size 100 \
    --learning_rate 1e-2 \
    --lr_decay 0.95 \
    --hidden_size 200 \
    --reg 1e-3 \
    --model_file ./asset/model.pkl > ./asset/train.log 2>&1 &