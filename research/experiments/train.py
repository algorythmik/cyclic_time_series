# This scirpot is a sample on how to keep track of models together with the
# makefile which takes care of data part. It is a work in progrress but
# independent of the rest of repo.
import argparse
import os
from functools import partial

import mlflow
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from toolbox import s3_utils

project_name = 'kistler-kv-dtu'
s3_bucket = ''  # insert s3 bucket name here.


n_classes = 5


def kfold_split(features, labels, n_splits):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_ids, test_ids in kfold.split(features, labels):
        yield (features[train_ids], features[test_ids],
               labels[train_ids], labels[test_ids])


def kfold_cross_validation(kfold_splits, model, epochs, seed=7):
    cvscores = []
    for x_train, x_test, y_train, y_test in kfold_splits:
        model_ = model()
        model_.fit(x_train, y_train, batch_size=64, epochs=epochs, verbose=0)
        scores = model_.evaluate(x_test, y_test, verbose=0)
        cvscores.append(scores)

    return cvscores


def build_model(n_inputs, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=5,
        kernel_size=50,
        strides=5,
        input_shape=(n_inputs, n_features))
    )
    model.add(tf.keras.layers.MaxPool1D(pool_size=20))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=20, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def load_data():
    data_dir = os.environ.get('DATA_DIR')
    df = pd.read_csv(os.path.join(data_dir, 'run_data', 'runs_data.csv'))
    df_feat = df.pivot(
        index='run_no',
        columns='time',
        values='demand_current')
    df_label = df.iloc[:, -2:].groupby(by='run_no').max()
    x = df_feat.values
    y = df_label.values - 1
    return x.reshape(-1, 512, 1), y


def train(**args):
    n_splits = 10
    x, y = load_data()
    model = partial(build_model, 512, 1)
    splits = kfold_split(x, y, n_splits)
    epochs = 1
    mlflow.log_param('epoch', epochs)
    cv_scores = kfold_cross_validation(splits, model, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--upload', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    args = vars(parser.parse_args())

    upload = args.pop('upload')
    if upload:
        s3_utils.repo_is_clean()

    with mlflow.start_run() as run:
        mlflow.log_params(args)
        train(**args)

    if upload:
        s3_utils.upload_run(run, s3_bucket, project_name)
