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

:'
for time in $(seq 0 21)
do
    for layer in fc7 conv5 conv3 conv1
    do
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/fdb_responses_2.hdf5
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/tnn_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/tnn_responses_2.hdf5
        python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/byp_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/byp_responses_2.hdf5
    done

    for layer in fc8 conv6 conv4 conv2
    do
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/fdb_responses.hdf5
        #python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/tnn_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/tnn_responses.hdf5
        python cal_catRDM.py --key ${layer}_${time} --savepath /mnt/fs0/chengxuz/Data/nd_response/byp_RDMs/RDM_${layer}_${time}.pkl --hdf5path /data/chengxuz/nd_response/byp_responses.hdf5
    done
done
'

#layer_now=fc_add_0
#python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs_obj/RDM_${layer_now}.pkl --hdf5path /data2/chengxuz/nd_response/temp_spa_responses_add.hdf5 --labelkey objid --labelfile /data2/chengxuz/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 0
#python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data2/chengxuz/nd_response/temp_spa_responses_add.hdf5 --labelkey objid --labelfile /data2/chengxuz/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1
#layer_now=fc12_0
#layer_now=fc_add1_0
#for layer_now in fc11 conv7 conv9 conv5 conv3 conv1
#for layer_now in fc11_0 conv7_0
#for layer_now in fc_add2_0 fc_add_0
#for layer_now in conv9_0 conv5_0 conv3_0 conv1_0
#for layer_now in fc7_21 conv5_21 conv3_21 conv1_21
#for layer_now in fc8_21 conv6_21 conv4_21 conv2_21
#for layer_now in fc_add1_0
#for layer_now in fc12_0 conv8_0 conv10_0 conv6_0
#for layer_now in conv4_0 conv2_0
for layer_now in conv1_0 conv2_0 conv3_0 conv4_0 conv5_0 fc6_0 fc_add_0
#for layer_now in fc7_0
do
    #python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data2/chengxuz/nd_response/temp_spa_responses.hdf5 --labelkey objid --labelfile /data2/chengxuz/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1 &
    #python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/temp_spa_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data2/chengxuz/nd_response/temp_spa_responses_2.hdf5 --labelkey objid --labelfile /data2/chengxuz/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1 &
    #python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data/chengxuz/nd_response/fdb_responses_2.hdf5 --labelkey objid --labelfile /mnt/fs0/chengxuz/Data/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1 &
    #python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/fdb_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data/chengxuz/nd_response/fdb_responses.hdf5 --labelkey objid --labelfile /mnt/fs0/chengxuz/Data/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1 &

    python cal_catRDM.py --key ${layer_now} --savepath /mnt/fs0/chengxuz/Data/nd_response/spatemp_sm2_RDMs_obj/RDM_${layer_now}.hdf5 --hdf5path /data2/chengxuz/nd_response/spatemp_sm2_responses.hdf5 --labelkey objid --labelfile /mnt/fs0/chengxuz/Data/nd_response/responses_otherlabels_2.hdf5 --numcat 9981 --writeway 1 & 
done
