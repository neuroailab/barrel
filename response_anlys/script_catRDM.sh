#for layer in conv1 conv2 conv3 conv4 conv5 fc6 fc7
#for layer in fc12 fc11 conv6 conv7 conv8 conv9 conv10 conv5 conv4 conv3 conv2 conv1
:'
for layer in fc11 conv7 conv9 conv5 conv3 conv1
do
    #python cal_catRDM.py --key ${layer}_${1} --savepath /mnt/fs0/chengxuz/Data/nd_response/RDM_${layer}_${1}.pkl
    python cal_catRDM.py --key ${layer}_${1} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs/RDM_${layer}_${1}.pkl --hdf5path /data2/chengxuz/nd_response/temp_spa_responses.hdf5
done

for layer in fc12 conv8 conv10 conv6 conv4 conv2
do
    #python cal_catRDM.py --key ${layer}_${1} --savepath /mnt/fs0/chengxuz/Data/nd_response/RDM_${layer}_${1}.pkl
    python cal_catRDM.py --key ${layer}_${1} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs/RDM_${layer}_${1}.pkl --hdf5path /data2/chengxuz/nd_response/temp_spa_responses_2.hdf5
done
'

for time in $(seq 0 21)
do
    for layer in fc7 conv5 conv3 conv1
    do
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/fdb_responses_2.hdf5
        python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/tnn_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/tnn_responses_2.hdf5
    done

    for layer in fc8 conv6 conv4 conv2
    do
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/fdb_responses.hdf5
        python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/tnn_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/tnn_responses.hdf5
    done
done
