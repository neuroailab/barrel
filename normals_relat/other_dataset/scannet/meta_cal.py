import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='The script to run the ext_cal_to_tfr.py script')
    parser.add_argument('--savedir', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/tmpdir', type = str, action = 'store', help = 'Path to the temporary directory hosting the depth and photo')
    parser.add_argument('--tfrdir', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/tfrecs/', type = str, action = 'store', help = 'Path to save the tfrecs, subfolders will be created')
    parser.add_argument('--exepath', default = '/om/user/chengxuz/barrel/ScanNet/SensReader/sens', type = str, action = 'store', help = 'Path for the SenseReader')
    parser.add_argument('--indxsta', default = 1, type = int, action = 'store', help = 'Start index of the sens files')
    parser.add_argument('--indxlen', default = 1, type = int, action = 'store', help = 'Length of indexes of the sens files')
    parser.add_argument('--scriptpath', default = '/om/user/chengxuz/barrel/barrel/normals_relat/other_dataset/scannet/ext_cal_to_tfr.py', type = str, action = 'store', help = 'Path to the script')
    parser.add_argument('--sensdir', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/scannet/', type = str, action = 'store', help = 'Path to the sens files')

    args    = parser.parse_args()

    sens_list = os.listdir(args.sensdir)

    for indx_sens in xrange(args.indxsta, min(args.indxlen + args.indxsta, len(sens_list))):
        now_sens = sens_list[indx_sens]

        sens_path = os.path.join(args.sensdir, now_sens)
        tfrname = 'scannet%i.tfrecord' % indx_sens

        cmd_str = 'python %s --tfrname %s --exepath %s --savedir %s --tfrdir %s --path %s' % (args.scriptpath, tfrname, args.exepath, os.path.join(args.savedir, 'd%i' % indx_sens), args.tfrdir, sens_path)
        
        os.system(cmd_str)

if __name__=="__main__":
    main()
