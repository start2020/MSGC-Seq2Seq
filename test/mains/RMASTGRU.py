import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from libs import para, main_common
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = para.common_para(parser)
    parser = para.RMASTGRU(parser)
    args = parser.parse_args()
    main_common.main(args)
