import numpy as np
import dldata.metrics.utils as utils
import dldata.stimulus_sets.hvm as hvm
import cPickle
import dldata.physiology.hongmajaj.mappings as mappings
import scipy.io as sio
import pdb
import h5py
import sys
import os
from sklearn.cross_validation import KFold
from sklearn import linear_model # for model_type = 0,1
from sklearn import cross_decomposition # for model_type = 2
from sklearn.metrics import r2_score
import argparse
import sklearn.decomposition
import gene_hvm_response
from scipy import misc

def resize_prep(img_unit8, want_out_w = 240, want_out_h = 320):
    return misc.imresize(img_unit8, [want_out_w, want_out_h])

def post_func_one(ret_val, which_one, want_indx = None):
    desire_one = ret_val[which_one]
    if desire_one is None:
        desire_one = ret_val[0]

    desire_one = desire_one.reshape(desire_one.shape[0], desire_one.size/desire_one.shape[0])
    if not want_indx == None:
        desire_one = desire_one[:, want_indx]
    return desire_one

def post_func(ret_val):
    ret_val_now = []
    for indx_now in xrange(len(ret_val)):
        if not ret_val[indx_now] is None:
            ret_val[indx_now] = ret_val[indx_now].reshape(ret_val[indx_now].shape[0], ret_val[indx_now].size/ret_val[indx_now].shape[0])
            ret_val_now.append(ret_val[indx_now])

    if len(ret_val_now)==1:
        return ret_val_now[0]
    else:
        return ret_val_now

class Feature_select_PCA:
    def __init__(self, datapath = '/mnt/data/imagenet2012.hdf5', source = 'train/images', nimgs = 5000, nfeats = 1500):
        self.file_in    = h5py.File(datapath, 'r')
        self.data       = self.file_in[source]
        self.nimgs      = nimgs
        self.nfeats     = nfeats

        self.curr_indx  = 0
        self.big_img_array = None

    def get_imagenet_imgs(self, preproc = resize_prep, want_out_w = 240, want_out_h = 320, num_channel = 3, kwargs = {}):
        # preproc takes one image and return the image preprocessed
        big_img_array = np.zeros([self.nimgs, want_out_w, want_out_h, num_channel])

        for tmp_indx in xrange(self.curr_indx, self.curr_indx + self.nimgs):
            if (tmp_indx - self.curr_indx) % 100==0:
                print('Now img indx: %i' % (tmp_indx - self.curr_indx))
            now_img     = np.asarray(self.data[tmp_indx])
            big_img_array[tmp_indx - self.curr_indx]    = preproc(now_img, want_out_w, want_out_h, **kwargs)

        big_img_array   = big_img_array.astype(np.float32)
        self.curr_indx  = self.curr_indx + self.nimgs
        self.big_img_array = big_img_array

    def get_PCA_ready(self, resp_func, kwargs, post_func, post_kwargs = {},  get_img_kwargs = {}):
        if self.big_img_array is None:
            self.get_imagenet_imgs(**get_img_kwargs)
        all_imgs    = self.big_img_array
        ret_func    = resp_func(all_imgs, **kwargs)
        img_resp    = post_func(ret_func, **post_kwargs)

        if isinstance(img_resp, list) and len(img_resp) > 1:
            self.pca = []
            for indx_res, res_now in enumerate(img_resp):
                tmp_pca     = sklearn.decomposition.PCA(n_components=self.nfeats)
                tmp_pca.fit(res_now)
                self.pca.append(tmp_pca)
        else:
            self.pca = sklearn.decomposition.PCA(n_components=self.nfeats)
            self.pca.fit(img_resp)

    def set_curr_indx(self, expect_val):
        self.curr_indx = expect_val

    def get_PCA_trans(self, to_trans, which_one=0):
        if isinstance(self.pca, list):
            return self.pca[which_one].transform(to_trans)
        else:
            return self.pca.transform(to_trans)

def main():
    parser = argparse.ArgumentParser(description='The script to generate responses of hvmdataset')
    parser.add_argument('--network', default = 0, type = int, action = 'store', help = '0 is alexnet, 1 is vgg')
    parser.add_argument('--layer', default = "1.1", type = str, action = 'store', help = 'which layer to extract')
    parser.add_argument('--loaddir', default = "/mnt/data/chengxuz/barrel/hvm_responses", type = str, action = 'store', help = 'where to store the file')
    parser.add_argument('--loadprefix', default = "hvm_layer_", type = str, action = 'store', help = 'Prefix of saving file')
    parser.add_argument('--savedir', default = "/home/chengxuz/barrel/barrel_github/normal_pretrained/fitting_results", type = str, action = 'store', help = 'where to store the file')
    parser.add_argument('--saveprefix', default = "regress_layer_", type = str, action = 'store', help = 'Prefix of saving file')
    parser.add_argument('--cachedir', default = "/home/chengxuz/barrel/barrel_github/normal_pretrained/fitting_cache", type = str, action = 'store', help = 'where to store the file')
    parser.add_argument('--dimethod', default = "random", type = str, action = 'store', help = "How to implement the dimension reduction, default is random, PCA available")
    parser.add_argument('--PCAMaxcontrol', default = -1, type = int, action = 'store', help = '<=0 means no control, else random sample that number of features')

    args    = parser.parse_args()

    os.system('mkdir -p ' + args.savedir)

    all_neuron_flag = 1

    cache_flag      = 1
    os.system('mkdir -p ' + args.cachedir)

    if args.network==0:
        no_double_out_list = ["1.1", "1.2", "1.3", "2.1", "3.1"]
        model_name = 'depthnormals_nyud_alexnet'
    else:
        no_double_out_list = ["1.1", "1.2", "1.3", "2.1"]
        model_name = 'depthnormals_nyud_vgg'

    if args.dimethod=='PCA':
        machine = gene_hvm_response.get_network(model_name)
        sel_pca = Feature_select_PCA()

    if args.layer in no_double_out_list:
        input_data_paths     = [os.path.join(args.loaddir, "%s%s.hdf5" % (args.loadprefix, args.layer))]
        save_file_paths      = [os.path.join(args.savedir, "%s%s.pkl" % (args.saveprefix, args.layer))]
        cache_file_paths     = [os.path.join(args.cachedir, "cache_%s%s.pkl" % (args.saveprefix, args.layer))]
    else:
        input_data_paths     = [os.path.join(args.loaddir, "%s%s_depths.hdf5" % (args.loadprefix, args.layer)),
                os.path.join(args.loaddir, "%s%s_normals.hdf5" % (args.loadprefix, args.layer))]
        save_file_paths      = [os.path.join(args.savedir, "%s%s_depths.pkl" % (args.saveprefix, args.layer)),
                os.path.join(args.savedir, "%s%s_normals.pkl" % (args.saveprefix, args.layer))]
        cache_file_paths     = [os.path.join(args.cachedir, "cache_%s%s_depths.pkl" % (args.saveprefix, args.layer)),
                os.path.join(args.cachedir, "cache_%s%s_normals.pkl" % (args.saveprefix, args.layer))]

    for input_data_path, save_file_path, cache_file_path in zip(input_data_paths, save_file_paths, cache_file_paths):

        sub_rep_time    = 10
        Subsample   = True
        #sample_num  = 1000

        # TODO make a sample_num choosing task, say: choose from 1000-2000
        sample_num  = 1500 # sample_num is the number of neurons we sampled from that layer

        tmp_fin         = h5py.File(input_data_path, 'r')
        if 'data' in tmp_fin:
            model_features  = tmp_fin['data']
        else:
            key_tmp         = tmp_fin.keys()
            key_tmp         = key_tmp[0]
            model_features  = tmp_fin[key_tmp]

        model_features  = model_features[0:model_features.shape[0]]
        model_features  = model_features.reshape(model_features.shape[0], model_features.size/model_features.shape[0])
        print(model_features.shape)

        dataset_hvm = hvm.HvMWithDiscfade()
        meta_hvm    = dataset_hvm.meta

        neural_fea  = dataset_hvm.neuronal_features
        if all_neuron_flag==0:
            indx_now    = dataset_hvm.IT_NEURONS
        else:
            indx_now    = range(neural_fea.shape[1])
        neural_fea  = neural_fea[:, indx_now]

        #Kfold   = 20
        #clf_alpha       = 1 
        clf_alpha       = 0.001
        n_com           = 25
        random_state    = 0
        Kfold           = 5
        ret_val         = []

        subsample_indx  = 0

        indsv3v6 = ((meta_hvm['var'] == 'V3') | (meta_hvm['var'] == 'V6'))

        IT_noise = dataset_hvm.noise_estimate(indsv3v6, 
                                        units=np.array(indx_now), 
                                        n_jobs=5)

        if cache_flag==1:
            cache_data = {}

        if args.dimethod=='PCA':
            sel_pca.set_curr_indx(0)
            which_one = 0
            if 'normals.hdf5' in input_data_path:
                which_one = 1

        #for sample_num in [500, 700, 1000, 1200, 1500, 1700]:
        for subsample_indx in xrange(sub_rep_time):
            kf      = KFold(neural_fea.shape[0], Kfold, shuffle = True, random_state = 0)
            if Subsample==True:
                #model_features_aftersub     = model_features[:, np.random.choice(model_features.shape[1], sample_num, 0)]
                if sample_num <= model_features.shape[1]:
                    if args.dimethod=='random':
                        tmp_indx_list   = np.random.choice(model_features.shape[1], sample_num, 0)
                        tmp_indx_list.sort()
                        model_features_aftersub     = model_features[:, tmp_indx_list]
                    elif args.dimethod=='PCA':
                        if args.PCAMaxcontrol>0:
                            if model_features.shape[1] >= args.PCAMaxcontrol:
                                tmp_indx_list   = np.random.choice(model_features.shape[1], args.PCAMaxcontrol, 0)
                                tmp_indx_list.sort()
                                model_features_inter     = model_features[:, tmp_indx_list]
                            else:
                                model_features_inter     = model_features
                                tmp_indx_list   = None
                        else:
                            tmp_indx_list   = None

                        sel_pca.get_imagenet_imgs()
                        sel_pca.get_PCA_ready(machine.infer_some_layer_depth_and_normals, kwargs = {'which_layer': args.layer}, post_func = post_func_one, post_kwargs = {'which_one': which_one, 'want_indx': tmp_indx_list})
                        if args.PCAMaxcontrol>0:
                            model_features_aftersub     = sel_pca.get_PCA_trans(model_features_inter)
                        else:
                            model_features_aftersub     = sel_pca.get_PCA_trans(model_features)
                    else:
                        raise Exception('Not implemented!')
                else:
                    model_features_aftersub     = model_features
                print(model_features_aftersub.shape, subsample_indx)

            predict_fea     = np.zeros(neural_fea.shape)

            for train, test in kf:
                #print(train.shape, test.shape)
                #print(train[0], test[0])
                #clf     = linear_model.Ridge(alpha=clf_alpha, random_state = random_state)
                #clf     = cross_decomposition.PLSRegression(n_components = 10)
                #for n_com in range(33, 45, 2):
                    #clf     = linear_model.Lasso(alpha=clf_alpha, random_state = random_state)
                clf     = cross_decomposition.PLSRegression(n_components = n_com, scale = False)

                now_train_data      = model_features_aftersub[train, :]
                now_train_label     = neural_fea[train, :]
                now_test_data       = model_features_aftersub[test, :]
                now_test_label      = neural_fea[test, :]
                #print(now_test_label.shape)
                clf.fit(now_train_data, now_train_label)

                new_test_label      = clf.predict(now_test_data)
                predict_fea[test]   = new_test_label
                #print(clf.score(now_train_data, now_train_label), clf.score(now_test_data, now_test_label), clf_alpha)
                print(clf.score(now_train_data, now_train_label), clf.score(now_test_data, now_test_label), n_com)
                sys.stdout.flush()

            
            #print(predict_fea.shape)

            unit_score  = r2_score(neural_fea, predict_fea, multioutput='raw_values')
            std_list    = np.var(neural_fea, 0)
            #print(std_list.shape)
            #print(np.sum(unit_score*std_list)/np.sum(std_list))
            #print(np.mean(unit_score))
            #print(r2_score(neural_fea, predict_fea, multioutput='variance_weighted'))
            new_score = unit_score/IT_noise[0]**2
            #print(np.median(new_score))
            #print(np.sum(new_score*std_list)/np.sum(std_list))
            #new_rsquare     = np.sum(new_score*std_list)/np.sum(std_list)
            new_rsquare     = np.median(new_score)
            
            if cache_flag==1:
                cache_data[subsample_indx]  = new_score

            if subsample_indx==0:
                ret_val     = [new_rsquare]
            else:
                ret_val[0]  = ret_val[0] + new_rsquare

        ret_val[0]  = ret_val[0]/sub_rep_time
        fout    = open(save_file_path, 'w')
        cPickle.dump(ret_val, fout)
        fout.close()

        if cache_flag==1:
            fout    = open(cache_file_path, 'w')
            cPickle.dump(cache_data, fout)
            fout.close()

if __name__=="__main__":
    main()

