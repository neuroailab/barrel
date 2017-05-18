xlen=10
ylen=10
filelen=125

for xstart in $(seq 0 ${xlen} ${filelen})
do
    for ystart in $(seq ${xstart} ${ylen} ${filelen})
    do
        sbatch script_partdismat.sh ${xstart} ${xlen} ${ystart} ${ylen}
    done
done

:'
xlen=1000
ylen=1000
whichlayer=1
#whichlayer=6
#whichlayer=5

#for whichlayer in 1 2 3 4
#for whichlayer in 5 6
for whichlayer in 4
do
    #for xstart in $(seq 0 ${xlen} 9981)
    for xstart in 1000 2000
    do
        #for ystart in $(seq ${xstart} ${ylen} 9981)
        for ystart in 4000
        do
            sbatch script_partRDM.sh ${xstart} ${xlen} ${ystart} ${ylen} ${whichlayer}
            #sbatch script_partRDM_om.sh ${xstart} ${xlen} ${ystart} ${ylen} ${whichlayer}
        done
    done
done
'

