:'
xlen=10
ylen=10

for xstart in $(seq 10 ${xlen} 119)
#for xstart in 0
do
    #for ystart in $(seq 0 ${ylen} 119)
    for ystart in $(seq ${xstart} ${ylen} 119)
    do
        sbatch script_partdismat.sh ${xstart} ${xlen} ${ystart} ${ylen}
    done
done
'

xlen=1000
ylen=1000
#whichlayer=1
#whichlayer=6
whichlayer=5

for xstart in $(seq 0 ${xlen} 9981)
#for xstart in 0
do
    for ystart in $(seq ${xstart} ${ylen} 9981)
    #for ystart in 0
    do
        sbatch script_partRDM.sh ${xstart} ${xlen} ${ystart} ${ylen} ${whichlayer}
    done
done
