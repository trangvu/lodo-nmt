#!/bin/bash
GAL_DIR='/home/trangvu/meta-al'
ROOT_DIR=$GAL_DIR
echo "Running on fitcluster"
module load python37
source $ROOT_DIR/env/bin/activate
python split.py