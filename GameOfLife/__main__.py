import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def train_gol():
    print("[GoL] Started Training")
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename="train.csv",
        target_dtype=np.int,
        features_dtype=np.float64
    )

    feature_columns = [tf.feature_column.numeric_column("x", shape=[training_set.data.shape[1]])]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10,10]
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True
    )

    # Train the model.
    classifier.train(input_fn=train_input_fn, steps=6000)
    print('[GoL] Training done on', training_set.data.shape[0], 'samples')

    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename="data.csv",
        target_dtype=np.int,
        features_dtype=np.float64
    )

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False
    )

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print(f"[GoL] Running on test data ({test_set.data.shape[0]} samples): Got {accuracy_score} accuracy")
    return classifier


def get_next_state(sample):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(sample)},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    return np.array([ON if int(prediction["classes"][0]) is 1 else OFF for prediction in predictions]).reshape(N, N)


# update the grid
def update(data):
    global grid
    main_sample = []
    for i in range(N):
        for j in range(N):
            sample = [
                grid[(i, j)],
                grid[i, (j-1)%N], grid[i, (j+1)%N],
                grid[(i-1)%N, j], grid[(i+1)%N, j],
                grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N],
            ]
            temp = list(map(lambda x: 1 if x == 255 else 0, sample))
            main_sample.append(temp)

    new_grid = get_next_state(main_sample)

    # update grid in animation
    mat.set_data(new_grid)
    grid = new_grid
    return [mat]


# Begin main script
N = 100
ON = 255
OFF = 0

# Train our classifier
classifier = train_gol()
vals = [ON, OFF]

# initial state: populate grid with random on/off
grid = np.random.choice(vals, N * N, p=[0.3, 0.7]).reshape(N, N)

# set up animation
fig, ax = plt.subplots()
mat = ax.matshow(grid)
ani = animation.FuncAnimation(fig, update)

# GO!
plt.show()
