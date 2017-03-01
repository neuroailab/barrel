#for layer in 1.1 1.2 1.3 2.1 2.2 2.3 2.4 2.5 2.6 3.1 3.2 3.3 3.4 3.5
#do
#    THEANO_FLAGS=device=gpu1 python gene_hvm_response.py --layer ${layer}
#done
#for layer in 1.1 1.2 1.3 2.1 2.2 2.3 2.4 2.5 2.6 3.1 3.2 3.3 3.4 3.5
for layer in 2.1 2.2 2.3 2.4 2.5 2.6 3.1 3.2 3.3 3.4 3.5
do
    #python neural_fit_response.py --layer ${layer}
    THEANO_FLAGS=device=gpu1 python neural_fit_response.py --layer ${layer} --dimethod PCA --saveprefix PCA_regress_layer_ --PCAMaxcontrol 150000
done
