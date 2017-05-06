suffix='0000.caffemodel'

for i in {1..10}
do
	echo '>>>>>>>>>>>>>>>>>>Testing '$i
	tools/test_net.py \
	    --gpu 0 \
	    --def models/pvanet/example_finetune/test.prototxt \
		--net output/faster_rcnn_pvanet/psdb_train/pvanet_frcnn_iter_$i$suffix \
	    --cfg models/pvanet/cfgs/submit_160715.yml \
		--imdb psdb_2015_test 
done

