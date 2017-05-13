for layer in conv1 conv2 conv3 conv4 conv5 fc6 fc7
do
    python cal_catRDM.py --key ${layer}_${1} --savepath /mnt/fs0/chengxuz/Data/nd_response/RDM_${layer}_${1}.pkl
done
