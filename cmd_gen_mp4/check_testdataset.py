import os

hdf_dir = "/scratch/users/chengxuz/barrel/barrel_relat_files/testdataset_diver"
bigsamnum = 48
seed_len = 251

for seed_now in xrange(seed_len):
    for now_samnum in xrange(bigsamnum):
        for name in ['teddy', 'duck']:
            now_path = os.path.join(hdf_dir, "Data%i_%i_%s_%i.hdf5" % (seed_now, now_samnum, name, seed_now))
            if not os.path.exists(now_path):
                print(now_path)

