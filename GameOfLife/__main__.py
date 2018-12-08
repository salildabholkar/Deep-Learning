import numpy as np
from numpy import array, int, float64, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.feature_column import numeric_column
from tensorflow.estimator import DNNClassifier, inputs

tf.logging.set_verbosity(tf.logging.ERROR)
ON = 255
OFF = 0
N = 100
train_file = 'train.csv'
test_file = 'data.csv'


class GameOfLife():
    def __init__(self, train_file, test_file, N):
        self.N = N
        self.grid = random.choice([ON, OFF], self.N * self.N, p=[0.3, 0.7]).reshape(self.N, self.N)
        self.classifier = None
        self.training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=train_file,
            target_dtype=int,
            features_dtype=float64
        )
        self.test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=test_file,
            target_dtype=int,
            features_dtype=float64
        )
        self.ani = None

    def train(self):
        print('[GoL] Started Training')
        self.classifier = DNNClassifier(
            feature_columns=[numeric_column('x', shape=[self.training_set.data.shape[1]])],
            hidden_units=[10, 10]
        )
        train_input_fn = inputs.numpy_input_fn(
            x={'x': array(self.training_set.data)},
            y=array(self.training_set.target),
            num_epochs=None,
            shuffle=True
        )
        self.classifier.train(input_fn=train_input_fn, steps=6000)
        print('[GoL] Training done on', self.training_set.data.shape[0], 'samples')

        test_input_fn = inputs.numpy_input_fn(
            x={'x': array(self.test_set.data)},
            y=array(self.test_set.target),
            num_epochs=1,
            shuffle=False
        )
        accuracy_score = self.classifier.evaluate(input_fn=test_input_fn)['accuracy']
        print(f'[GoL] Running on test data ({self.test_set.data.shape[0]} samples): Accuracy {accuracy_score}')
        return self.grid

    def get_next_state(self, sample):
        predict_input_fn = inputs.numpy_input_fn(
            x={'x': array(sample)},
            num_epochs=1,
            shuffle=False
        )
        predictions = list(self.classifier.predict(input_fn=predict_input_fn))
        return array([ON if int(prediction['classes'][0]) is 1 else OFF for prediction in predictions]).reshape(self.N,
                                                                                                                self.N)

    def get_neighbors(self, i, j):
        return [self.grid[a % self.N, b % self.N] for a in range(i - 1, i + 2) for b in range(j - 1, j + 2)]

    def update(self, data):
        main_sample = []
        for i in range(self.N):
            for j in range(self.N):
                sample = self.get_neighbors(i, j)
                temp = list(map(lambda x: 1 if x == 255 else 0, sample))
                main_sample.append(temp)

        new_grid = self.get_next_state(main_sample)
        mat.set_data(new_grid)
        self.grid = new_grid
        return [mat]


if __name__ == '__main__':
    GoL = GameOfLife(train_file, test_file, N)
    grid = GoL.train()
    fig, ax = plt.subplots()
    mat = ax.matshow(grid)
    ani = animation.FuncAnimation(fig, GoL.update)
    plt.show()
