:'
for objindx in 0 1
#for objindx in 0
do
    for oindxsta in $(seq 0 14)
    #for oindxsta in 0
    do
        for spindxsta in $(seq 0 13)
        #for spindxsta in 0
        do
            #sbatch script_sbatch_hdf5_om.sh ${objindx} ${oindxsta} ${spindxsta}
            sh run_check_hdf5.sh ${objindx} ${oindxsta} ${spindxsta}
        done
    done
done
'

#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 1 --scindxlen 1 --oindxsta ${2} --pindxsta ${3}
#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s_val --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 3 --scindxlen 3 --oindxsta ${2} --pindxsta ${3} --generatemode 2 --testmode 2

for objindx in 0 1
#for objindx in 0
do
    for pindxsta in 1 $(seq 12 19)
    #for pindxsta in 1
    do
        for scindxsta in $(seq 0 5)
        #for scindxsta in 0
        do
            for oindxsta in $(seq 0 3 14)
            #for oindxsta in 0
            do
                sbatch script_sbatch_hdf5_om.sh ${objindx} ${pindxsta} ${scindxsta} ${oindxsta}
            done
        done
    done
done
