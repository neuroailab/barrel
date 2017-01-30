for k in 100 1000 5000 10000 50000 100000
do
    sbatch --job-name=hyper${k} script_sbatch.sh ${k}
done
