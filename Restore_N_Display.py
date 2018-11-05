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
init = tf.global_variables_initializer()
sess = tf.Session()


model = {}
# model['W1'] = tf.get_variable("w1",shape=[D,H])
# model['W2'] = tf.get_variable("w2",shape=[H,1])
saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'TF_RL_save6000.meta'))



with tf.Session() as sess:
    saver.restore(sess, os.path.join(os.getcwd(), 'TF_RL_save6000'))
    # print sess.run('w1:0')
    # print sess.run('w2:0')


    graph = tf.get_default_graph()
    model['W1'] = graph.get_tensor_by_name('w1:0')
    model['W2'] = graph.get_tensor_by_name('w2:0')

    # print model['W1']
    #
    h_fc1 = tf.nn.relu(tf.matmul(observations,model['W1']))
    h_fc2 = tf.matmul(h_fc1,model['W2'])
    probability = tf.nn.sigmoid(h_fc2)


    # In[7]:

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None
    xs,ys,drs=[],[],[]
    running_reward = None
    reward_sum=0
    episode_number=1



    won = 0
    lost = 0


    # In[8]:

    Monitor(env, '/root/tmp/pong-sub-1',force=True)
    #env.reset()
    #print "Reset Done!!"


    # In[ ]:

    while True:


        env.render()

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
            observation = env.reset() # reset env
            prev_x = None

    observation = env.reset()
    sess.close()
