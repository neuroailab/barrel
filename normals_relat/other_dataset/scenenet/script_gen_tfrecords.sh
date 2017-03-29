loaddir=/mnt/data2/chengxuz/scenenet/train
savedir=/mnt/data2/chengxuz/scenenet/train_tfrecords

for k in $(seq 2 16)
#for k in 0
#for k in 1
do
    #python save_to_tfrecords.py --mode 0 --path ${loaddir}/${k} --savedir ${savedir}/${k}/photo
    #python save_to_tfrecords.py --mode 1 --path ${loaddir}/${k} --savedir ${savedir}/${k}/depth
    #python save_to_tfrecords.py --mode 2 --path ${loaddir}/${k} --savedir ${savedir}/${k}/instance
    sbatch --job-name=tfrecs${k} script_normal_tfrecs.sh ${k}
done
