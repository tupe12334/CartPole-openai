import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras_visualizer import visualizer


def main():
    epochs = 10
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    model = build_model(states, actions)
    model.summary()
    for epoch in range(1, epochs):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = random.choice([0, 1])
            n_state, reward, done, info = env.step(action)
            score += reward
            print(f'Epoch:{epoch}, score:{score}')


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation="linear"))
    visualizer(model, format='png', view=True)
    return model


if __name__ == '__main__':
    main()
