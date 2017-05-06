suffix='0000.caffemodel'

for i in {1..20}
do
	echo '>>>>>>>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/full5/test.pt \
		--net output/pvanet_full/psdb_train/pvanet_frcnn_iter_$i$suffix \
	    --cfg models/pvanet/cfgs/submit_160715_full.yml \
		--imdb psdb_2015_test 
done

