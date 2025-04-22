#!/bin/bash
LOCAL_TRAIN_FOLDER=$(pwd)
IMAGES_PATH=$(pwd)/SHOES/IMATGES_HDPE

docker run --gpus all -it --name=train_deep -v ${LOCAL_TRAIN_FOLDER}:/home/deep/ \
	-v ${IMAGES_PATH}:/home/deep/imgs/ -p 8888:8888 \
	-p 6006:6006 --rm picvisa/deep-gen:v3 \
	/bin/bash -c "(python3 /home/deep/SHOES/similarity_v2.py &) && (tensorboard --logdir=/home/deep/runs/ --port=6006 --bind_all)"
