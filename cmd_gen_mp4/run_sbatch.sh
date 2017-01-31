#for k in 100 1000 5000
#for k in 100 500 1000
#do
#    sbatch --job-name=dnhyper${k} script_sbatch.sh ${k}
#done

#sbatch --job-name=dhmons200 script_sbatch_monserver.sh 200
for k in $(seq 1 10)
do
    sbatch --job-name=dhmonc200 script_sbatch_monclient.sh 23333 test_db
done
