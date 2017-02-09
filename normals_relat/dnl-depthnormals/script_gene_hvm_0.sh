for layer in 1.1 1.2 1.3 2.1 2.2 2.3 2.4 2.5 2.6 3.1 3.2 3.3 3.4 3.5
do
    THEANO_FLAGS=device=gpu2 python gene_hvm_response.py --layer ${layer} --network 1 --saveprefix hvm_vgg_layer_
    python neural_fit_response.py --layer ${layer} --network 1 --loadprefix hvm_vgg_layer_ --saveprefix regress_vgg_layer_
done