from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


class SpaceInvader:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def build_model(cls, height, width, channels, actions):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (8, 8),
                strides=(4, 4),
                activation="relu",
                input_shape=(3, height, width, channels),
            )
        )
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(actions, activation="linear"))
        return cls(model)

    def build_agent(self, actions):
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr="eps",
            value_max=1.0,
            value_min=0.1,
            value_test=0.2,
            nb_steps=10000,
        )
        memory = SequentialMemory(limit=2000, window_length=3)
        return DQNAgent(
            model=self.model,
            memory=memory,
            policy=policy,
            enable_dueling_network=True,
            dueling_type="avg",
            nb_actions=actions,
            nb_steps_warmup=1000,
        )
