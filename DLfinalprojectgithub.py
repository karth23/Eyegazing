import numpy as np,scipy,matplotlib
from pylab import *
import os
import tensorflow as tf
############################################################
#Loading data
os.chdir("../Deeplearning/Finalproject")
npzfile = np.load("train_and_val.npz")
train_eye_left = npzfile["train_eye_left"]
train_eye_right = npzfile["train_eye_right"]
train_face = npzfile["train_face"]
train_face_mask = npzfile["train_face_mask"]
train_y = npzfile["train_y"]
val_eye_left = npzfile["val_eye_left"]
val_eye_right = npzfile["val_eye_right"]
val_face = npzfile["val_face"]
val_face_mask = npzfile["val_face_mask"]
val_y = npzfile["val_y"]
bs = 200
###############################################################################
#####Functions to be used
def batch
(train_eye_left,train_eye_right,train_face,train_face_mask,train_y,overall_size,name="batch"):
	with tf.name_scope(name):
		randomlist = np.random.randint(overall_size,size=bs)
	return train_eye_left[randomlist],train_eye_right[randomlist],train_face[randomlist],train_face_mask[randomlist],train_y[randomlist]
def valbatch(val_eye_left,val_eye_right,val_face,val_face_mask,val_y,overall_size):
	randomlist = np.random.randint(overall_size,size=bs)
	return val_eye_left[randomlist],val_eye_right[randomlist],val_face[randomlist],val_face_mask[randomlist],val_y[randomlist]
def errorfunc(pred,actual):
	error = 0;loss=0
	for i in range(len(pred)):
	error = error+(((pred[i][0]-actual[i][0])**2)+((pred[i][1]-actual[i][1])**2))**0.5
	loss = loss + ((pred[i][0]-actual[i][0])**2)+((pred[i][1]-actual[i][1])**2)
	err = error/len(pred)
	loss = np.mean(loss**0.5)
	return err,loss
trainerrorarray=[];iterarray=[];valerrorarray=[];costarray = []
####################################################################################
#Placeholder
x_eye_left_ = tf.placeholder(tf.float32,[None,64,64,3],name="x_eye_left_")
x_eye_right_ = tf.placeholder(tf.float32,[None,64,64,3],name="x_eye_right_")
x_face_ = tf.placeholder(tf.float32,[None,64,64,3],name="x_face_")
x_mask_ = tf.placeholder(tf.float32,[None,25,25],name="x_mask_")
y_ = tf.placeholder(tf.float32,[None,2],name="y_")
####################################################################################
########## Variables used for eye pathways
w1 = tf.get_variable("w1",shape =[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable("w2",shape =[5,5,32,32],initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable("w3",shape =[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
w4 = tf.get_variable("w4",shape =[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
w5= tf.get_variable("w5",shape=[576,256],initializer=tf.contrib.layers.xavier_initializer())
w6 = tf.get_variable("w6",shape=[512,256],initializer=tf.contrib.layers.xavier_initializer())
w10 = tf.Variable(tf.zeros([60,60,32]),name="w10")
w20 = tf.Variable(tf.zeros([26,26,32]),name="w20")
w30 = tf.Variable(tf.zeros([11,11,32]),name="w30")
w40 = tf.Variable(tf.zeros([3,3,64]),name="w40")
w50 = tf.Variable(tf.zeros([1,256]),name="w50")
w60 = tf.Variable(tf.zeros([1,256]),name="w60")
#########################
############### Variables used for face pathway
w1_face = tf.get_variable("w1_face",shape =[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer())
w2_face = tf.get_variable("w2_face",shape =[5,5,32,32],initializer=tf.contrib.layers.xavier_initializer())
w3_face = tf.get_variable("w3_face",shape =[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
w4_face = tf.get_variable("w4_face",shape=[7744,256],initializer=tf.contrib.layers.xavier_initializer())
w5_face = tf.get_variable("w5_face",shape=[256,512],initializer=tf.contrib.layers.xavier_initializer())
w10_face = tf.Variable(tf.zeros([60,60,32]),name="w10_face")
w20_face = tf.Variable(tf.zeros([26,26,32]),name="w20_face")
w30_face = tf.Variable(tf.zeros([11,11,64]),name="w30_face")
w40_face = tf.Variable(tf.zeros([1,256]),name="w40_face")
w50_face = tf.Variable(tf.zeros([1,512]),name="w50_face")
#########################
############### Variables used for mask pathway
w1_mask = tf.get_variable("w1_mask",shape =
[625,256],initializer=tf.contrib.layers.xavier_initializer())
w2_mask = tf.get_variable("w2_mask",shape =
[256,128],initializer=tf.contrib.layers.xavier_initializer())
w10_mask = tf.Variable(tf.zeros([1,256]),name="w10_mask")
w20_mask = tf.Variable(tf.zeros([1,128]),name="w20_mask")
w1_eye_face_mask = tf.get_variable("w1_eye_face_mask",shape =
[896,256],initializer=tf.contrib.layers.xavier_initializer())
w10_eye_face_mask = tf.get_variable("w10_eye_face_mask",shape =
[1,256],initializer=tf.contrib.layers.xavier_initializer())
w2_eye_face_mask = tf.get_variable("w2_eye_face_mask",shape =
[256,2],initializer=tf.contrib.layers.xavier_initializer())
w20_eye_face_mask = tf.Variable(tf.zeros([1,2]),name="w20_eye_face_mask")
#############################################################################################
############################### Tensorflow operations
##################################
################## Tensorflow operations for eye; The architecture involves 4convolution layers each for left eye and right eye, 3 pooling layers in between the convolution layers, and finally 2 fully connected layers
conv1_eye_left = tf.nn.conv2d(x_eye_left_, w1, strides = [1,1,1,1],
padding="VALID",name="conv1_eye_left")
conv1_eye_left = tf.add(conv1_eye_left,w10,name="conv1_eye_left_withbias")
conv1_eye_left = tf.nn.relu(conv1_eye_left,name="relu_conv1_eye_left")
network_eye_left = tf.nn.max_pool(conv1_eye_left,ksize=[1, 2, 2, 1],strides=[1, 2, 2,1], padding='VALID',name="pool_eye_left_1")
conv1_eye_right = tf.nn.conv2d(x_eye_right_, w1, strides = [1,1,1,1],
padding="VALID",name="conv1_eye_right")
conv1_eye_right = tf.add(conv1_eye_right,w10,name="conv1_eye_rightwithbias")
conv1_eye_right = tf.nn.relu(conv1_eye_right,name="relu_conv1_eye_right")
network_eye_right = tf.nn.max_pool(conv1_eye_right,ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='VALID',name="pool_eye_right_1")

conv2_eye_left = tf.nn.conv2d(network_eye_left, w2,strides = [1,1,1,1],padding="VALID",name="conv2_eye_left")
conv2_eye_left = tf.add(conv2_eye_left,w20,name="conv2_eye_leftwithbias")
conv2_eye_left = tf.nn.relu(conv2_eye_left,name="relu2_eye_left")
network2_eye_left = tf.nn.max_pool(conv2_eye_left,ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='VALID',name="pool_eye_left_2")
conv2_eye_right = tf.nn.conv2d(network_eye_right, w2,strides = [1,1,1,1],padding="VALID",name="conv2_eye_right")
conv2_eye_right = tf.add(conv2_eye_right,w20,name="conv2_eye_rightwithbias")
conv2_eye_right = tf.nn.relu(conv2_eye_right,name="relu2_eye_right")
network2_eye_right = tf.nn.max_pool(conv2_eye_right,ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='VALID',name="pool2_eye_right")
conv3_eye_left = tf.nn.conv2d(network2_eye_left, w3,strides = [1,1,1,1],padding="VALID",name="conv3_eye_left")
conv3_eye_left = tf.add(conv3_eye_left,w30,name="conv3_eye_leftwithbias")
conv3_eye_left = tf.nn.relu(conv3_eye_left,name="relu3_eye_left")
network3_eye_left = tf.nn.max_pool(conv3_eye_left,ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='VALID',name="pool3_eye_left")
conv4_eye_left = tf.nn.conv2d(network3_eye_left, w4,strides = [1,1,1,1],padding="VALID",name="conv4_eye_left")
conv4_eye_left = tf.add(conv4_eye_left,w40,name="conv4_eye_leftwithbias")
conv4_eye_left = tf.nn.relu(conv4_eye_left,name="relu4_eye_left")
conv3_eye_right = tf.nn.conv2d(network2_eye_right, w3,strides = [1,1,1,1],padding="VALID",name="conv3_eye_right")
conv3_eye_right = tf.add(conv3_eye_right,w30,name="conv3_eye_rightwithbias")
conv3_eye_right = tf.nn.relu(conv3_eye_right,name="relu3_eye_right")
network3_eye_right = tf.nn.max_pool(conv3_eye_right,ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='VALID',name="pool3_eye_left")
conv4_eye_right = tf.nn.conv2d(network3_eye_right, w4,strides = [1,1,1,1],padding="VALID",name="conv4_eye_right")
conv4_eye_right = tf.add(conv4_eye_right,w40,name="conv4_eye_rightwithbias")
conv4_eye_right = tf.nn.relu(conv4_eye_right,name="relu4_eye_right")
reshaped_eye_left = tf.reshape(conv4_eye_left,[bs,576],name="reshaped_eye_left")
#reshaped_eye_left = tf.reshape(conv3_eye_left,[bs,7744])
fc_eye_left_matmul = tf.matmul(reshaped_eye_left,w5,name="fc_eye_left_matmul")
fc_eye_left = tf.add(fc_eye_left_matmul,w50,name="fc_eye_left")
reshaped_eye_right = tf.reshape(conv4_eye_right,[bs,576],name="reshaped_eye_right")
fc_eye_right_matmul = tf.matmul(reshaped_eye_right,w5,name="fc_eye_right_matmul")
fc_eye_right = tf.add(fc_eye_right_matmul,w50,name="fc_eye_right")
fc_eye = tf.concat([fc_eye_left,fc_eye_right],1,name="fc_eye")
fc_eye_2_matmul = tf.matmul(fc_eye,w6,name="fc_eye_2_matmul")
fc_eye_2 = tf.add(fc_eye_2_matmul,w60,name="fc_eye_layer2")
##################################
################## Tensorflow operations for face. The architecture for the facepathway involves 3 convolutional layers and 2 pooling layers and finally 2 fully connected layers
conv1_face = tf.nn.conv2d(x_face_, w1_face, strides = [1,1,1,1],padding="VALID",name="conv1_face")
conv1_face = tf.add(conv1_face,w10_face,name="conv1_facewithbias")
conv1_face = tf.nn.relu(conv1_face,name="relu1_face")
network_face = tf.nn.max_pool(conv1_face,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name="pool1_face")
conv2_face = tf.nn.conv2d(network_face, w2_face, strides = [1,1,1,1],padding="VALID",name="conv2_face")
conv2_face = tf.add(conv2_face,w20_face,name="conv2_facewithbias")
conv2_face = tf.nn.relu(conv2_face,name="relu2_face")
network2_face = tf.nn.max_pool(conv2_face,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name="pool2_face")
conv3_face = tf.nn.conv2d(network2_face, w3_face,strides = [1,1,1,1],padding="VALID",name="conv3_face")
conv3_face = tf.add(conv3_face,w30_face,name="conv3_facewithbias")
conv3_face = tf.nn.relu(conv3_face,name="relu3_face")
reshaped_face = tf.reshape(conv3_face,[bs,7744],name="reshaped_face")
fc_face_matmul = tf.matmul(reshaped_face,w4_face,name="fc_face_matmul")
fc_face = tf.add(fc_face_matmul,w40_face,name="fc1_face")
fc_face2_matmul = tf.matmul(fc_face,w5_face,name="fc2_face_matmul")
fc_face_2 = tf.add(fc_face2_matmul,w50_face,name="fc2_face")
##################################
################## Connecting fully connected layers of eye and face
connected_eye_face = tf.concat([fc_eye_2,fc_face_2],1,name="connected_eye_face")
##################################
################## Tensorflow operations for face mask. The architecture of the mask involves 2 fully connected layers
reshape_mask = tf.reshape(x_mask_,[bs,625],name="reshaped_mask")
fc_mask_1 = tf.add(tf.matmul(reshape_mask,w1_mask),w10_mask,name="fc1_mask")
fc_mask_2 = tf.add(tf.matmul(fc_mask_1,w2_mask),w20_mask,name="fc2_mask")
##################################
################## Connecting fully connected layers of eye , face, mask
fc_eye_face_mask = tf.concat([connected_eye_face,fc_mask_2],1,name="fc_eye_face_mask")
fc_eye_face_mask_2 = tf.add(tf.matmul(fc_eye_face_mask,w1_eye_face_mask),w10_eye_face_mask,name="fc2_eye_face_mask")
###########################
# Tensorflow operations at y node and optimizer
z = tf.add(tf.matmul
(fc_eye_face_mask_2,w2_eye_face_mask),w20_eye_face_mask,name="prediction")
ypred = tf.Variable(tf.zeros([bs,2]),name="ypred")
yactual = tf.Variable(tf.zeros([bs,2]),name="yactual")
predictionassign = ypred.assign(z)
actualassign = yactual.assign(y_)
cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum((z - y_)**2)),name="cost")
train_op =tf.train.AdamOptimizer(0.001,0.9,name="optimizer").minimize(cost)


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for iter in range(10001):
	tr_eye_left,tr_eye_right,tr_face,tr_face_mask,tr_y = batch(train_eye_left,train_eye_right,train_face,train_face_mask,train_y,48000)
	sess.run(fc_eye_left,feed_dict={x_eye_left_:tr_eye_left})
	sess.run(fc_eye_right,feed_dict={x_eye_right_:tr_eye_right})
	sess.run(fc_eye_2,feed_dict={x_eye_left_:tr_eye_left,x_eye_right_:tr_eye_right})
	sess.run(connected_eye_face,feed_dict={x_eye_left_:tr_eye_left,x_eye_right_:tr_eye_right,x_face_:tr_face})
	sess.run(fc_mask_2,feed_dict={x_mask_:tr_face_mask})
	sess.run(y_,feed_dict={y_:tr_y})
	sess.run(actualassign,feed_dict={y_:tr_y})
	sess.run(train_op,feed_dict={x_eye_left_:tr_eye_left,x_eye_right_:tr_eye_right,x_face_:tr_face,x_mask_:tr_face_mask,y_:tr_y})
	writer = tf.summary.FileWriter("graph")
	writer.add_graph(sess.graph)
	if (iter%50==0):
	sess.run(predictionassign,feed_dict={x_eye_left_:tr_eye_left,x_eye_right_:tr_eye_right,x_face_:tr_face,x_mask_:tr_face_mask})
	trainerror,trainloss = errorfunc(ypred.eval(),yactual.eval())
	trainerrorarray.append(trainerror)
	iterarray.append((iter*bs/48000.0))
	costarray.append(trainloss)
	v_eye_left,v_eye_right,v_face,v_face_mask,v_y = valbatch(val_eye_left,val_eye_right,val_face,val_face_mask,val_y,5000)
	sess.run(fc_eye_left,feed_dict={x_eye_left_:v_eye_left})
	sess.run(fc_eye_right,feed_dict={x_eye_right_:v_eye_right})
	sess.run(fc_eye_2,feed_dict={x_eye_left_:v_eye_left,x_eye_right_:v_eye_right})
	sess.run(connected_eye_face,feed_dict={x_eye_left_:v_eye_left,x_eye_right_:v_eye_right,x_face_:v_face})
	sess.run(fc_mask_2,feed_dict={x_mask_:v_face_mask})
	sess.run(predictionassign,feed_dict={x_eye_left_:v_eye_left,x_eye_right_:v_eye_right,x_face_:v_face,x_mask_:v_face_mask})
	sess.run(y_,feed_dict={y_:v_y})
	sess.run(actualassign,feed_dict={y_:v_y})
	valerror,valloss = errorfunc(ypred.eval(),yactual.eval())
	valerrorarray.append(valerror)

	print 'iter',iter,'trainloss',trainloss,'trainerror',trainerror, 'val error',valerror

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(iterarray,costarray, 'b-')
	ax2.plot(iterarray,trainerrorarray, 'r-', iterarray,valerrorarray, 'g-')
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('cost')
	ax1.set_ylim([0.0,100.0])
	ax2.set_ylabel('accuracy')
	ax2.set_yticks(np.arange(1.0,5.0,0.2))
	ax2.set_ylim([1.0,5.0])
	savefig("accuracies_combined_morelayers2.png")
	plt.show()
	tf.get_collection("validation_nodes")
	tf.add_to_collection("validation_nodes",x_eye_left_)
	tf.add_to_collection("validation_nodes",x_eye_right_)
	tf.add_to_collection("validation_nodes",x_face_)
	tf.add_to_collection("validation_nodes",x_mask_)
	tf.add_to_collection("validation_nodes",z)
	saver = tf.train.Saver()
	save_path = saver.save(sess, "my_model")
