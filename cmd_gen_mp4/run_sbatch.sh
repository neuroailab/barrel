#for k in 100 1000 5000
#for k in 100 500 1000
#do
#    sbatch --job-name=dnhyper${k} script_sbatch.sh ${k}
#done

k=400
#sbatch --job-name=dhmons${k} script_sbatch_monserver.sh ${k} exp4
sbatch --job-name=dhmons${k} script_sbatch_monserver.sh ${k} exp5
sleep 5
for k in $(seq 1 15)
do
    sbatch --job-name=dhmonc${k} script_sbatch_monclient.sh 23333 test_db
done
