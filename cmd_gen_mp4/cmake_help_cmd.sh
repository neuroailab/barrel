# on om7
cmake -D BULLET_PHYSICS_SOURCE_DIR:SPRING=/om/user/chengxuz/barrel/bullet3 -D BULLET_ROOT:SPRING=/om/user/chengxuz/usr/local /om/user/chengxuz/barrel/barrel/bullet_demos_extracted/examples/

# on neuroaicluster node1
cmake -D BULLET_PHYSICS_SOURCE_DIR:SPRING=/mnt/fs0/chengxuz/barrel_relate/bullet3 -D BULLET_ROOT:SPRING=/mnt/fs0/chengxuz/lib_relat/bullet /mnt/fs0/chengxuz/barrel/bullet_demos_extracted/examples

# on mac, run cmd_hdf5
python cmd_hdf5.py --pathhdf5 /Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s --pathexe /Users/chengxuz/barrel/bullet/example_build/ExampleBrowser/App_ExampleBrowser --fromcfg /Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/opt_results/para_ --pathconfig /Users/chengxuz/barrel/bullet/barrle_related_files/configs --generatemode 2
