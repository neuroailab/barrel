#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s_new_name --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --pindxlen 36 --scindxlen 6 --oindxsta ${2} --spindxsta ${3} --generatemode 2 --testmode 2 --checkmode 1
#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s_val --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs_2 --objindx /om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd/04256520/1037fd31d12178d396f164a988ef37cc/1037fd31d12178d396f164a988ef37cc.obj --generatemode 3 --testmode 2 --hdf5suff 1037fd31d12178d396f164a988ef37cc --smallolen 2
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000

#python cmd_to_tfrecords.py --objsta ${1} --objlen ${2} --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords --infodir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_info --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --seedbas 10000
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000 --checkmode 0

#python cmd_dataset.py --objsta 4039 --bigsamnum 12 --savedir /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --loaddir /Users/chengxuz/barrel/bullet/bullet3/data --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_
#python cmd_dataset.py --objsta 4039 --bigsamnum 12 --savedir /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --loaddir /Users/chengxuz/barrel/bullet/bullet3/data --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_
python cmd_dataset.py --objsta 241 --bigsamnum 12 --savedir /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --loaddir /Users/chengxuz/barrel/bullet/bullet3/data --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_
