nohup python -u main.py --mode test \
    --data_path data/cifar-10-batches-py \
    --model_file ./asset/model.pkl > ./asset/test.log 2>&1 &
