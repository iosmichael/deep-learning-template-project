# here's an example of train model that is responsible for the whole pipeline.
import os
import sys
import random
import datetime
import time
import glob
import shutil
import logging
import argparse
import yaml


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


parser = argparse.ArgumentParser("ResNet20-cifar100")
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')  # 8192
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='init learning rate')  
parser.add_argument('--config', help="configuration file",
                    type=str, default="configs/meta.yml")
parser.add_argument('--save_dir', type=str,
                    help="save exp floder name", default="exp1")
args = parser.parse_args()

# process argparse & yaml
if not args.config:
    opt = vars(args)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(args)
    args = opt
else:  # yaml priority is higher than args
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)

args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime("%mM_%dD_%HH") + "_" + \
    "{:04d}".format(random.randint(0, 1000))

# 文件处理
if not os.path.exists(os.path.join("exp", args.exp_name)):
    os.makedirs(os.path.join("exp", args.exp_name))


# 日志文件
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")

fh = logging.FileHandler(os.path.join("exp", args.exp_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)

# 配置文件
with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
    yaml.dump(args, f)

# Tensorboard文件
writer = SummaryWriter("exp/%s/runs/%s-%05d" %
                       (args.exp_name, time.strftime("%m-%d", time.localtime()), random.randint(0, 100)))

# 文件备份
create_exp_dir(os.path.join("exp", args.exp_name),
               scripts_to_save=glob.glob('*.py'))