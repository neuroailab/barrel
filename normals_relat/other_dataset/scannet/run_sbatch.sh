len_data=10

#for k in $(seq 0 ${len_data} 1524)
#for k in 0
for k in $(seq ${len_data} ${len_data} 1524)
do
    sbatch  --job-name=scannet${k} script_cal.sh ${k} ${len_data}
done
