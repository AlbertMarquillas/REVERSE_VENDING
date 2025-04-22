#!/bin/bash
LOCAL_TRAIN_FOLDER=$(pwd)
IMAGES_PATH=$(pwd)/PROVA_TEST

docker run --gpus all -it --name=test_deep -v ${LOCAL_TRAIN_FOLDER}:/home/deep/ \
	-v ${IMAGES_PATH}:/home/deep/imgs/ -p 9999:8888 \
	-p 6007:6006 --rm picvisa/deep-gen:v4.1 \
	/bin/bash -c "(python3 /home/deep/test.py &) && (tensorboard --logdir=/home/deep/runs/ --port=6006 --bind_all)"
