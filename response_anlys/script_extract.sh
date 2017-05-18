#key_tmp=conv1
#for key_tmp in conv3 conv5
for key_tmp in conv2 conv4 conv6
do
    for time in $(seq 0 21)
    do
        #python extract_ave_resp.py --check 1 --key ${key_tmp}_${time} --savename ${key_tmp}.hdf5
        python extract_ave_resp.py --check 1 --key ${key_tmp}_${time} --savename ${key_tmp}.hdf5 --hdf5path /data/chengxuz/nd_response/resp_spa_temp_1.hdf5
    done
done
