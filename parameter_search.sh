nohup python -u main.py --mode hyper_search \
    --data_path data/cifar-10-batches-py \
    --num_epochs 25 \
    --batch_size 100 \
    --lr_list 0.01,0.005 \
    --hidden_list 200,100,50 \
    --reg_list 0.001,0.0001 \
    --lr_decay 0.95 \
    --model_file best_model.pkl > ./asset/hyperparameter_search.log 2>&1 &
