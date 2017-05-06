tools/train_net.py  \
    --gpu 0 \
    --solver models/pvanet/full5/solver.prototxt \
	--weights models/pvanet/full5/original.model \
    --iters 200000 \
    --cfg models/pvanet/cfgs/trainFull.yml \
    --imdb psdb_2015_train  &> log_psdb &

