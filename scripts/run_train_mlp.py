import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from diploma.models.neural_nets import MultiLayerPerceptron
from diploma.utils.trainer import train_model
from diploma.utils.datasets import PharmaDataset

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df_train_LOX = pd.read_csv('data/training_set_LOX.csv')
    df_train_ANT = pd.read_csv('data/training_set_antioxidant.csv')
    df_train_LOX = df_train_LOX.drop(columns=['Title'], axis=1)
    df_train_ANT = df_train_ANT.drop(columns=['Title'], axis=1)

    mean_values_LOX = df_train_LOX.mean()
    mean_values_ANT = df_train_ANT.mean()

    df_train_LOX = df_train_LOX.fillna(mean_values_LOX)
    df_train_ANT = df_train_ANT.fillna(mean_values_ANT)

    frames = [df_train_LOX, df_train_ANT]
    train_set = pd.concat(frames)

    x = train_set.loc[:, train_set.columns != 'class'].to_numpy()
    y = train_set.iloc[:, 0].to_numpy()
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y,
        test_size=0.2,
        random_state=0
    )

    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)

    train_dataset = PharmaDataset(features=x_train, labels=y_train)
    val_dataset = PharmaDataset(features=x_validation, labels=y_validation)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4
    )

    model = MultiLayerPerceptron(
        hidden_layers=[128, 256, 512],
        input_dim=x_train.shape[1],
        output_dim=2,
    )
    optimizer = Adam(
        params=model.parameters(),
        lr=1e-03
    )
    metrics = train_model(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        optimizer=optimizer,
        loss_functions=[BCEWithLogitsLoss()],
        epochs=10,
        evaluation_steps=25,
        best_model_path='models/nn.pt'
    )

    steps = metrics.pop('steps')
    for metric in metrics.values():
        plt.plot(steps, metric)
    plt.grid()
    plt.title('Training procedure')
    plt.xlabel('Step')
    plt.legend(metrics.keys())
    plt.savefig('results/losses', dpi=500)

    df = pd.read_csv('data/test_set.csv').drop(columns=['Title'], axis=1)
    test = df.to_numpy()
    test = scaler.transform(test)
    probs = model.predict(test)
    np.set_printoptions(suppress=True)
    print(probs)
