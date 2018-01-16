import tensorflow as tf
import numpy as np

import itertools
import better_exceptions
import gym
from gym import wrappers
from tqdm import tqdm

import tf_util
import load_policy

slim = tf.contrib.slim

BATCH_SIZE = 32
LEARNING_RATE = 0.001
BETA = 0.9

class Policy():

	def __init__(self, env, obs_samples=None):

		if obs_samples is None:
			obs_samples = np.array([env.observation_space.sample() for _ in range(1000)])

		self.obs_mean = obs_samples.mean(axis=0)
		self.obs_std = obs_samples.std(axis=0)

		self.state = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))
		self.target_action = tf.placeholder(tf.float32, [None] + list(env.action_space.shape))

		normalized = (self.state - self.obs_mean) / self.obs_std

		net = slim.fully_connected(normalized, 50, 
								   scope='fc1', activation_fn=tf.nn.relu)
		net = slim.fully_connected(net, 50, 
								   scope='fc2', activation_fn=tf.nn.relu)
		self.policy = slim.fully_connected(net, env.action_space.shape[0], 
										   scope='policy', activation_fn=None)

		# L2-loss
		self.loss = tf.reduce_mean(tf.reduce_sum((self.policy-self.target_action)**2, axis=1))

		optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA)

		self.train_op = optimizer.minimize(self.loss)

	def predict(self, state):
		sess = tf.get_default_session()
		return sess.run(self.policy, feed_dict={self.state: state})

	def update(self, state, action):
		sess = tf.get_default_session()
		loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.state: state, self.target_action: action})
		return loss

	def test_run(self, env, max_steps):
		observations = []
		actions = []
		rewards = 0.

		state = env.reset()

		for step in itertools.count():
			observations.append(state)
			actions.append(self.predict(np.expand_dims(state, axis=0))[0])

			next_state, reward, done, _ = env.step(actions[-1])

			state = next_state
			rewards += reward

			if step >= max_steps or done:
				break

		experience = {'observations': np.stack(observations, axis=0),
					  'actions': np.squeeze(np.stack(actions, axis=0)),
					  'reward': rewards}

		return experience


def gather_expert_experience(num_rollouts, 
							 env, 
							 policy_fn, 
							 max_steps):

	with tf.Session():
		tf_util.initialize()

		returns = []
		observations = []
		actions = []

		for i in tqdm(range(num_rollouts)):
			state = env.reset()
			done = False
			rewards = 0.
			steps = 0

			while not done:
				action = policy_fn(state[None,:])
				observations.append(state)
				actions.append(action)

				next_state, reward, done, _ = env.step(action)

				state = next_state
				rewards += reward
				steps += 1

				if steps >= max_steps:
					break

			returns.append(rewards)

		expert_data = {'observations': np.stack(observations, axis=0),
					   'actions': np.squeeze(np.stack(actions, axis=0)),
					   'returns': np.array(returns)}

		return expert_data


def behavior_cloning(env_name=None,
					 expert_policy_file=None,
					 num_rollouts=10,
					 max_timesteps=None,
					 num_epochs=100,
					 save=None):
	
	tf.reset_default_graph()

	env = gym.make(env_name)
	max_steps = max_timesteps or env.spec.timestep_limit

	print('[BA] Loading and building expert policy')
	expert_policy_fn = load_policy.load_policy(expert_policy_file)

	print('[BA] Gather experience...')
	data = gather_expert_experience(num_rollouts, env, expert_policy_fn, max_steps)

	print('[BA] Expert\'s reward mean: {:4f}({:4f})'.format(np.mean(data['returns']), np.std(data['returns'])))

	print('[BA] Building cloning policy')
	policy = Policy(env, data['observations'])

	with tf.Session():
		tf_util.initialize()

		for epoch in tqdm(range(num_epochs)):
			num_samples = data['observations'].shape[0]
			perm = np.random.permutation(num_samples)

			obs_samples = data['observations'][perm]
			action_samples = data['actions'][perm]

			loss = 0.
			for k in range(0, obs_samples.shape[0], BATCH_SIZE):
				loss += policy.update(obs_samples[k:k+BATCH_SIZE], action_samples[k:k+BATCH_SIZE])

			new_exp = policy.test_run(env, max_steps)
			tqdm.write('[BA] Epoch {:3d}, Loss {:4f}, Reward {:4f}'.format(epoch, loss/num_samples, new_exp['reward']))

		if save is not None:
			env = wrappers.Monitor(env, save, force=True)

		results = []
		for _ in tqdm(range(num_rollouts)):
			results.append(policy.test_run(env, max_steps)['reward'])

		print('[BA] Reward mean & std of cloned policy: {:4f}({:4f})'.format(np.mean(results), np.std(results)))

	return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)


def dagger(env_name=None,
		   expert_policy_file=None,
		   num_rollouts=10,
		   max_timesteps=None,
		   num_epochs=100,
		   save=None):
	
	tf.reset_default_graph()

	env = gym.make(env_name)
	max_steps = max_timesteps or env.spec.timestep_limit

	print('[DA] Loading and building expert policy')
	expert_policy_fn = load_policy.load_policy(expert_policy_file)

	print('[DA] Gather experience...')
	data = gather_expert_experience(num_rollouts, env, expert_policy_fn, max_steps)

	print('[DA] Expert\'s reward mean: {:4f}({:4f})'.format(np.mean(data['returns']), np.std(data['returns'])))

	print('[DA] Building cloning policy')
	policy = Policy(env, data['observations'])

	with tf.Session():
		tf_util.initialize()

		for epoch in tqdm(range(num_epochs)):
			num_samples = data['observations'].shape[0]
			perm = np.random.permutation(num_samples)

			obs_samples = data['observations'][perm]
			action_samples = data['actions'][perm]

			loss = 0.

			for k in range(0, obs_samples.shape[0], BATCH_SIZE):
				loss += policy.update(obs_samples[k:k+BATCH_SIZE], action_samples[k:k+BATCH_SIZE])

			new_exp = policy.test_run(env, max_steps)

			# Data aggregation steps
			# Supervision signal comes from expert policy
			new_exp_len = new_exp['observations'].shape[0]

			expert_expected_actions = []

			for k in range(0, new_exp_len, BATCH_SIZE):
				expert_expected_actions.append(expert_policy_fn(new_exp['observations'][k:k+BATCH_SIZE]))

			# Added new experience into original one
			data['observations'] = np.concatenate((data['observations'], new_exp['observations']), axis=0)
			data['actions'] = np.concatenate([data['actions']] + expert_expected_actions, axis=0)
			tqdm.write('[DA] Epoch {:3d}, Loss {:4f}, Reward {:4f}'.format(epoch, loss/num_samples, new_exp['reward']))

		if save is not None:
			env = wrappers.Monitor(env, save, force=True)

		results = []
		for _ in tqdm(range(num_rollouts)):
			results.append(policy.test_run(env, max_steps)['reward'])
		print('[DA] Reward mean & std of cloned policy with DAGGER: {:4f}({:4f})'.format(np.mean(results), np.std(results)))

	return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)


if __name__ == "__main__":

    import os
    env_models = [('Ant-v1','experts/Ant-v1.pkl'),
                  ('HalfCheetah-v1','experts/HalfCheetah-v1.pkl'),
                  ('Hopper-v1','experts/Hopper-v1.pkl'),
                  ('Humanoid-v1','experts/Humanoid-v1.pkl'),
                  ('Reacher-v1','experts/Reacher-v1.pkl'),
                  ('Walker2d-v1','experts/Walker2d-v1.pkl'),]

    results = []

    for env, model in env_models :

        ex_mean, ex_std, bc_mean, bc_std = behavior_cloning(env_name=env,
                                                            expert_policy_file=model,
                                                            save=os.path.join(os.getcwd(), env, 'bc'))

        _, _, da_mean, da_std = dagger(env_name=env,
                       		           expert_policy_file=model,
                                	   num_epochs=40,
                                	   save=os.path.join(os.getcwd(), env, 'da'))

        results.append((env, ex_mean, ex_std,
        			    bc_mean, bc_std,
        			    da_mean, da_std))

    for env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std in results :
        print('Env: {}, Expert: {:4f}({:4f}), Behavior Cloning: {:4f}({:4f}), Dagger: {:4f}({:4f})'.format(env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))
 