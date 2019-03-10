import numpy as np 
import matplotlib.pyplot as plt

from helpers import conv2d,add_bias,add_bias1,meanpool2,fconnect,activ_fun

test_im = np.load('images/testim2.npy')
plt.imshow(np.reshape(test_im,(28,28)))
plt.show()

wc1 = np.load('../../Predictor/server_net/models/model_16/weights/weights1.npy')
wc2 = np.load('../../Predictor/server_net/models/model_16/weights/weights2.npy')
wf1 = np.load('../../Predictor/server_net/models/model_16/weights/weights3.npy')
otw = np.load('../../Predictor/server_net/models/model_16/weights/weights4.npy')

bc1 = np.load('../../Predictor/server_net/models/model_16/biases/bc1.npy')
bc2 = np.load('../../Predictor/server_net/models/model_16/biases/bc2.npy') 
bf1 = np.load('../../Predictor/server_net/models/model_16/biases/bd1.npy')
otb = np.load('../../Predictor/server_net/models/model_16/biases/bout.npy')

test_im = np.reshape(test_im,(1,28,28,1))


#conv2d takes (1,28,28,1) image and (5,5,1,5) kernel and gives (1,28,28,5) output
con1 = conv2d(test_im,wc1)
# np.save('con1.npy',con1)
print "conv1"
print con1
#add_bias takes the 5 (1,28,28) inputs and 5 biases, and adds the bias to every pixel in the 5 inputs
con1 = add_bias(con1,bc1)
# np.save('con1bias.npy',con1)
print "conv1addbias"
print con1
#activ_fun takes (1,28,28,5) input and squares each and every element in the input
con1 = activ_fun(con1)
# np.save('con1act.npy',con1)
print "conv1act"
print con1
#meanpool2 takes (1,28,28,5) input and performs meanpooling on each of the 5 28x28 matrices seperately
#and gives a (1,14,14,5) output
mean1 = meanpool2(con1)
# np.save('con1mean.npy',mean1)
print "mean pooling 1"
print mean1
#conv2d takes (1,14,14,5) input and (5,5,5,10) kernel and gives (1,14,14,10) output
con2 = conv2d(mean1,wc2)
# np.save('con2.npy',con2)
print "conv2"
print con2
#add_bias takes the 10 (1,14,14) inputs and 10 biases, and adds the bias to every pixel in the 10 inputs
con2 = add_bias(con2,bc2)
# np.save('con2bias.npy',con2)
print "con2addbias"
print con2
#activ_fun takes (1,14,14,10) input and squares each and every element in the input
#con2 = activ_fun(con2)
print "con2act"
print con2
#meanpool2 takes (1,14,14,10) input and performs meanpooling on each of the 10 14x14 matrices seperately
#and gives a (1,7,7,10) output
mean2 = meanpool2(con2)
# np.save('con2mean.npy',mean2)
print "mean 2"
print mean2
#we flattern out the (1,7,7,4) matrix into one long 1d array
fc1 = np.reshape(mean2,7*7*bc2.shape[0])
#convert the 7*7*10 array into 1024 array by fully connecting using (3136,1024) weights
fc1 = fconnect(fc1,wf1)
# np.save('fc1.npy',fc1)
print "fully connected 1"
print fc1
#add 1024 biases to the 1024 outputs
fc1 = add_bias1(fc1,bf1)
# np.save('fc1bias.npy',fc1)
print "fully connected add bias"
print fc1
#square the 1024 outputs
fc1 = activ_fun(fc1)
# np.save('fc1act.npy',fc1)
print "fc1 act"
print fc1
#convert 1024 nodes into 10 nodes by fully connection using (1024,10) weights
fc2 = fconnect(fc1,otw)
# np.save('fc2.npy',fc2)
print "fully connect 2"
print fc2
#add 10 biases to the 10 weights
logits = add_bias1(fc2,otb)
# np.save('fc2bias.npy',logits)
print "logits - fc2 add bias"

print logits
print np.argmax(logits)../../Predictor/server_net/models/