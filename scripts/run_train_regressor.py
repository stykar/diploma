import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from diploma.models.neural_nets import ConvolutionalNetwork
from diploma.utils.datasets import PharmaRegressionDataset
from diploma.utils.regression_trainer import train_model


if __name__ == "__main__":


    df = pd.read_csv('data/lox_dpph_regression_pchembl.csv')
    df = df.drop(columns=['Title'], axis=1)

    mean_values = df.mean()
    df = df.fillna(mean_values)

    x = df.loc[:, df.columns != 'pchembl'].to_numpy()
    y = df.loc[:, 'pchembl'].to_numpy().reshape(len(df), 1)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y,
        test_size=0.2,
        random_state=0
    )

    train_data = np.concatenate((x_train, y_train), axis=1)
    validation_data = np.concatenate((x_validation, y_validation), axis=1)
    # We do this to scale labels as well
    scaler = preprocessing.MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_validation_data = scaler.transform(validation_data)
    x_train = scaled_train_data[:, :-1]
    y_train = scaled_train_data[:, -1].reshape(x_train.shape[0], 1)
    #print(y_train)
    x_validation = scaled_validation_data[:, :-1]
    y_validation = scaled_validation_data[:, -1].reshape(x_validation.shape[0], 1)

    train_dataset = PharmaRegressionDataset(features=x_train, labels=y_train)
    val_dataset = PharmaRegressionDataset(features=x_validation, labels=y_validation)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32
    )

    model = ConvolutionalNetwork(
        output_dim=1,
        dropout_prob=0.0
    )
    print(model)
    print(f'model trainable parameters: {model.calculate_params()}')
    optimizer = Adam(
        params=model.parameters(),
        lr=1e-03
    )
    metrics = train_model(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        optimizer=optimizer,
        loss_functions=[MSELoss()],
        epochs=20,
        best_model_path='models/nn_regression.pt'
    )

    steps = metrics.pop('epochs')
    for metric_name in ['loss', 'mean_abs_error']:
        plt.figure()
        plt.plot(steps, metrics[f'train_{metric_name}'])
        plt.plot(steps, metrics[f'val_{metric_name}'])
        plt.grid()
        plt.ylabel(metric_name)
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'])
        plt.savefig(f'results/conv_{metric_name}_regression', dpi=500)

    
    new_features = [f'{feature}'.replace(' ', '_') for feature in df.columns]
    final_features = []
    df_test = pd.read_csv('data/test_set.csv').drop(columns=['Title'], axis=1)

    for feature in new_features:
        for old_feature in df_test.columns:
            if feature in old_feature:
                final_features.append(old_feature)
                break

    df_test = df_test[final_features]
    df_test['dummy'] = 0
    test = df_test.to_numpy()
    test = scaler.transform(test)
    test = test[:, :-1]
    values = model.predict(test)
    np.set_printoptions(suppress=True)
    print(values)
