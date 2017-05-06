tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/example_finetune/solver.prototxt \
	--weights psdbVeh_150000.caffemodel \
    --iters 100000 \
    --cfg models/pvanet/cfgs/train.yml \
    --imdb psdb_2015_train  &> log_psdb &

