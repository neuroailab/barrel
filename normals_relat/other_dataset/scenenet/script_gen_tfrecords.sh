loaddir=/mnt/data2/chengxuz/train
savedir=/mnt/data2/chengxuz/train_tfrecords

for k in $(seq 0 16)
do
    python save_to_tfrecords.py --mode 0 --path ${loaddir}/${k} --savedir ${savedir}/${k}/photo
    python save_to_tfrecords.py --mode 1 --path ${loaddir}/${k} --savedir ${savedir}/${k}/depth
    python save_to_tfrecords.py --mode 2 --path ${loaddir}/${k} --savedir ${savedir}/${k}/instance
done
