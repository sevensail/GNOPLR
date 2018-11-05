import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf
from gym import wrappers
from gym.wrappers import Monitor

import os
print "Running this from: ",os.getcwd()

H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
resume = False
render = False

image_size = 80
D = image_size*image_size

observations = tf.placeholder(tf.float32,[None,D],name='input_x')
#x_image = tf.reshape(observation, [-1,image_size,image_size,1])

#Required Functions
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

#network definition
model = {}
model['W1'] = tf.get_variable("w1",shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
model['W2'] = tf.get_variable("w2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
#layer-1
h_fc1 = tf.nn.relu(tf.matmul(observations,model['W1']))
h_fc2 = tf.matmul(h_fc1,model['W2'])
probability = tf.nn.sigmoid(h_fc2)

print observations.get_shape()
print model['W1'].get_shape()
print h_fc1.get_shape()
print model['W2'].get_shape()
print probability.get_shape()


#learning setup
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1],name='input_y')
advantages = tf.placeholder(tf.float32,name='reward_signal')
loss = tf.reduce_mean((tf.square(input_y-probability)*advantages))
newGrads = tf.gradients(loss,tvars)

#Optimizer initialization
adam = tf.train.AdamOptimizer(learning_rate=0.001)
W1Grad = tf.placeholder(tf.float32,name="batch_gradW1")
W2Grad = tf.placeholder(tf.float32,name="batch_gradW2")
batchGrad = [W1Grad,W2Grad]

updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


# In[7]:

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs,ys,drs=[],[],[]
running_reward = None
reward_sum=0
episode_number=1

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
gradBuffer = sess.run(tvars)

for ix in range(len(gradBuffer)): gradBuffer[ix] = 0.0 * gradBuffer[ix]

saver = tf.train.Saver()

won = 0
lost = 0


# In[8]:

Monitor(env, '/root/tmp/pong-sub-1',force=True)
#env.reset()
#print "Reset Done!!"


# In[ ]:

while episode_number <= 10000:

    if episode_number%50 == 0:
        render = True
    else:
        render = False


    if render: env.render()

    cur_x = prepro(observation)
    x = cur_x-prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    x = x.reshape([1,D])

    tfProb = sess.run(probability,feed_dict={observations: x})
    action = 2 if np.random.uniform() < tfProb else 3

    xs.append(x)
    y = 1 if action == 2 else 0
    ys.append(y)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if len(drs) > 3000:
        drs.pop(0)
        ys.pop(0)
        xs.pop(0)

    if reward !=0:
        if reward >0: won+=1
        else: lost+=1

    if done:
        print "Episode ",episode_number," Completed and took ",len(xs)," steps"
        episode_number += 1
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        xs,ys,drs = [],[],[]

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
        for ix,grad in enumerate(tGrad): gradBuffer[ix] += grad

        if episode_number % batch_size == 0:
            sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
            for ix in range(len(gradBuffer)): gradBuffer[ix] = 0.0 * gradBuffer[ix]

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % 10 == 0: print 'resetting env. episode %d reward total was %f. running mean: %f, won: %.1f %%' % (episode_number, reward_sum, running_reward, 100.0*float(won)/float(won+lost))
        if episode_number % 1000 == 0: save_path = saver.save(sess,  os.path.join(os.getcwd(), 'TF_RL_save'+str(episode_number)))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

observation = env.reset()
sess.close()
