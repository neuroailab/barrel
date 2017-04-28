import argparse


parser = argparse.ArgumentParser(description='The script to train the catenet for barrel')
parser.add_argument('--pathconfig', default = "catenet_config.cfg", type = str, action = 'store', help = 'Path to config file')

args = parser.parse_args()
print(args)

args2 = parser.parse_args(args.pathconfig.split())
print(args2)
