#loaddir=/mnt/data2/chengxuz/scenenet/train
loaddir=/mnt/data2/chengxuz/scenenet_val/val
#savedir=/mnt/data2/chengxuz/scenenet/train_tfrecords
savedir=/mnt/data2/chengxuz/scenenet_val/val_tfrecords

#for k in $(seq 2 16)
#for k in 0
#for k in 1
#do
    #python save_to_tfrecords.py --mode 0 --path ${loaddir}/${k} --savedir ${savedir}/${k}/photo
    #python save_to_tfrecords.py --mode 1 --path ${loaddir}/${k} --savedir ${savedir}/${k}/depth
    #python save_to_tfrecords.py --mode 2 --path ${loaddir}/${k} --savedir ${savedir}/${k}/instance
    #sbatch --job-name=tfrecs${k} script_normal_tfrecs.sh ${k}
    #sbatch --job-name=tfrecs${k} script_normal_tfrecs.sh
#done

file_len=30

for key in photo instance depth normal
#for key in photo
do
    for indx_sta in $(seq 0 ${file_len} 843)
    #for indx_sta in 0
    do
        sbatch  --job-name=tfrecs${indx_sta} script_combine_tfrecs.sh ${key} ${indx_sta} ${file_len}
    done
done
