for objindx in 0 1
#for objindx in 0
do
    for oindxsta in $(seq 0 3)
    #for oindxsta in 0
    do
        for pindxsta in $(seq 0 3)
        #for pindxsta in 0
        do
            sbatch script_sbatch_hdf5_om.sh ${objindx} ${oindxsta} ${pindxsta}
        done
    done
done


#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 1 --scindxlen 1 --oindxsta ${2} --pindxsta ${3}
