import cmd_gen
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to run the hyperopt')
    parser.add_argument('--neval', default = 100, type = int, action = 'store', help = 'Number of eval_num')

    args    = parser.parse_args()

    cmd_gen.do_hyperopt(args.neval)
