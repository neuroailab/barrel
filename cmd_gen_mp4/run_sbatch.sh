#for k in 100 1000 5000
for k in 100 500 1000
do
    sbatch --job-name=dnhyper${k} script_sbatch.sh ${k}
done
