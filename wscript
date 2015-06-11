import maf
import maflib.util
import maflib.rules
import os
import random
import subprocess
import shutil
import Image
import caffe
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import string
import caffe.proto.caffe_pb2
import google.protobuf.text_format

def options(opt):
	pass

def configure(conf):
	pass

def build(exp):
	root = "data/"
	exp(source=[],
		target=["resized_images"],
		parameters=[{"resize_size":224}],
		rule=resize_images(root=root))

	exp(source=["resized_images"],
		target=["train-test.txt"],
		rule=make_lists())

	exp(source=["train-test.txt"],
	    target=["train.txt", "test.txt"],
	    parameters=[{"train_ratio":.5}],
	    rule=simple_split_dataset(seed=109100))
t
	exp(source=["train.txt"],
		target="train_db",
		rule='GLOG_logtostderr=1 convert_imageset.bin -backend leveldb / ${SRC[0].abspath()} ${TGT}')

	exp(source=["test.txt"],
		target="test_db",
		rule='GLOG_logtostderr=1 convert_imageset.bin -backend leveldb / ${SRC[0].abspath()} ${TGT}')

	exp(source="train_db",
		target="mean.binaryproto",
		rule="compute_image_mean.bin -backend leveldb ${SRC} ${TGT}")

	prototxt = "../train_val.prototxt"
	exp(source=[prototxt, "train_db", "test_db", "mean.binaryproto"],
		target="train_and_test.prototxt",
		rule=configure_data_layer(train_batchsize=32, test_batchsize=32))

	max_iter = 6000
	exp(source='train_and_test.prototxt',
	            target='solver.prototxt',
	            parameters=maflib.util.product({
	                'base_lr': [0.00001],
	                'momentum': [0.9],
	                'weight_decay': [0.004],
	                'lr_policy': ["fixed"],
	                # 'gamma': [0.1],
	                # 'stepsize': [2000],
	                'max_iter': [max_iter],
	                'snapshot': [2000],
	                'solver_mode':[1]
	            }),
	            rule=create_solver(
	                test_iter=30, test_interval=200, display=100))
	
	exp(source=["solver.prototxt"],
		target=["log.txt", "snapshots", 'final_model'],
		rule=caffe_train(max_iter=max_iter))

	# # for finetuning
	# exp(source=["solver.prototxt", "../blvc_alexnet.caffemodel"],
	# 	target=["log.txt", "snapshots", 'final_model'],
	# 	rule=caffe_train(max_iter=max_iter))


@maflib.util.rule
def resize_images(task):
	i = 0
	path = task.parameter["root"]
	print path
	print os.path.abspath(path)

	dst_dir = task.outputs[0].abspath()
	if os.path.isdir(dst_dir) == False:
		os.makedirs(dst_dir)

	size = task.parameter['resize_size']
	for root, dirs, files in os.walk(path):
		for dir in dirs:
			print dir
			dst_dir_class = os.path.join(dst_dir, dir)
			if os.path.isdir(dst_dir_class) == False:
				os.makedirs(dst_dir_class)
			for root2, dirs2, files2 in os.walk(os.path.join(path, dir)):
				for file in files2:
					root3, ext = os.path.splitext(file)
					if ext == '.jpg' or ext == '.JPG' or ext == '.png' or ext == '.PNG':
						
						try:
							img = Image.open(os.path.join(root2, file))
						except:
							print str(file) + "can't open!"
							continue

						file = file.replace(' ', '')
						if img.mode != "RGB":
							img = img.convert("RGB")
						img.resize((size, size)).save(dst_dir+"/"+dir+"/"+file) 
						i += 1
						if i % 100 == 0:
							print str(i) + "now.."


@maflib.util.rule
def add_mirrored_file(task):
	input_dir = task.inputs[0].abspath()
	output_dir = task.outputs[0].abspath()
	try:
		os.makedirs(output_dir)
	except:
		pass

	r = re.compile('-[0-9]+')
	for root, dirs, files in os.walk(input_dir):
		for file_ in files:
			original = Image.open(os.path.join(root, file_))
			mirror = original.transpose(Image.FLIP_LEFT_RIGHT)

			p = r.search(file_)
			reverse_angle = str(360 - int(file_[p.start()+1:p.end()])).zfill(3)
			filename_m = output_dir + '/' + os.path.splitext(file_)[0][0:p.start()+1] + reverse_angle + "m.jpg"
			print filename_m
			mirror.save(filename_m)
			original.save(output_dir+'/'+ file_)

@maflib.util.rule
def make_lists(task):
	#param
	pre_file_list = []
	tr_list = []
	val_list = []
	test_list = []
	dir_list = []
	
	path = task.inputs[0].abspath()
	for root, dirs, files in os.walk(path):
		for dir in dirs:
			print dir
			dir_list.append(dir)

	dir_list.sort()

	class_num = 0
	for dir in dir_list:
		path2 = os.path.join(path, dir)
		print path2
		for root, dirs, files in os.walk(path2):
			for file_ in files:
				print file_
				filename = os.path.join(root, file_)
				pre_file_list.append([filename, class_num])

		class_num += 1

	random.shuffle(pre_file_list)

	list_file = open(task.outputs[0].abspath(), 'w')
	
	for instance in pre_file_list:
		list_file.write(instance[0] + " " + str(instance[1]) + "\n")

@maflib.util.rule
def caffe_train(task):
	output_dir = task.outputs[1].abspath()

	try:
		os.makedirs(output_dir)
	except:
		pass

	f = open(task.inputs[0].abspath(), "r")
	base_solver = f.read()
	f.close()

	solver = output_dir + "/solver.prototxt"
	f = open(solver, "w")
	f.write(base_solver)
	f.write("snapshot_prefix: \"%s/\" \n" % output_dir)
	f.close()

	envs = dict(os.environ)
	envs["GLOG_logtostderr"] = str(1)


	# For finetuning
	# subprocess.check_call(["caffe.bin", "train", "-solver", solver, "-weights", task.inputs[1].abspath()], env=envs,
	# 	stderr=open(task.outputs[0].abspath(), "w"))
	subprocess.check_call(["caffe.bin", "train", "-solver", solver], env=envs,
		stderr=open(task.outputs[0].abspath(), "w"))


	shutil.copyfile(output_dir+'/_iter_{}.caffemodel'.format(task.parameter["max_iter"]), task.outputs[2].abspath())

@maflib.util.rule
def simple_split_dataset(task):
    if "seed" in task.parameter:
        random.seed(task.parameter["seed"])
    else:
        random.seed(time.clock())

    data = []
    with open(task.inputs[0].abspath()) as labels_file:
        for line in labels_file:
            data.append(line)

    random.shuffle(data)
    train_size = int(len(data) * task.parameter["train_ratio"])
    train = data[:train_size]
    test = data[train_size:]

    with open(task.outputs[0].abspath(), 'w') as out_train:
        for line in train:
            out_train.write(line)
    with open(task.outputs[1].abspath(), 'w') as out_test:
        for line in test:
            out_test.write(line)


@maflib.util.rule
def configure_data_layer(task):
    net_str = task.inputs[0].read()
    net = caffe.proto.caffe_pb2.NetParameter()
    google.protobuf.text_format.Merge(net_str, net) 

    for i in xrange(2):
        target_layer = net.layer[i]
        data_layer = target_layer.data_param

        data_layer.source = task.inputs[i+1].abspath()

        if len(task.inputs) >= 4:
            data_layer.mean_file = task.inputs[3].abspath()

        if 'crop_size' in task.parameter:
            data_layer.crop_size = task.parameter['crop_size']
        if 'mirror' in task.parameter:
            data_layer.mirror = task.parameter['mirror']
        if 'shuffle' in task.parameter:
            data_layer.shuffle = task.parameter['shuffle']

    if 'train_batchsize' in task.parameter:
        net.layer[0].data_param.batch_size = task.parameter['train_batchsize']

    if 'test_batchsize' in task.parameter:
        net.layer[1].data_param.batch_size = task.parameter['test_batchsize']

    result = google.protobuf.text_format.MessageToString(net)
    task.outputs[0].write(result)

@maflib.util.rule
def create_solver(task):
    solver = caffe.proto.caffe_pb2.SolverParameter()
    solver.net = task.inputs[0].abspath()

    for key in task.parameter:
        if hasattr(solver, key):
            if key == 'lr_change_at':
                for p in task.parameter[key]:
                    getattr(solver, key).append(p)
            elif key == 'test_iter':
                getattr(solver, key).append(task.parameter[key])
            else:
                setattr(solver, key, task.parameter[key])

    task.outputs[0].write(
    google.protobuf.text_format.MessageToString(solver))


