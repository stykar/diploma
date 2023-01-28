import pandas as pd
from torch.utils.data import DataLoader
from diploma.models.neural_nets import MultiLayerPerceptron
from diploma.utils.trainer import train_model
from diploma.utils.datasets import PharmaDataset

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df_train_LOX = pd.read_csv('../data/training_set_LOX.csv')
    df_train_ANT = pd.read_csv('../data/training_set_antioxidant.csv')
    df_train_LOX = df_train_LOX.drop(columns = ['Title'], axis = 1)
    df_train_ANT = df_train_ANT.drop(columns = ['Title'], axis = 1)

    mean_values_LOX = df_train_LOX.mean()
    mean_values_ANT = df_train_ANT.mean()

    df_train_LOX = df_train_LOX.fillna(mean_values_LOX)
    df_train_ANT = df_train_ANT.fillna(mean_values_ANT)

    frames = [df_train_LOX, df_train_ANT]
    train_set = pd.concat(frames)

    x = train_set.loc[:, train_set.columns != 'class'].to_numpy()
    y = train_set.iloc[:,0].to_numpy()
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=0)

    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)

    train_loader = DataLoader(dataset=PharmaDataset(
        features=x_train,
        labels=y_train
    ))
    validation_loader = DataLoader(dataset=PharmaDataset(
        features=x_validation,
        labels=y_validation
    ))

    model = 
    optimizer = 
    loss_function = 

    train_model(

    )