import time
import torch
import os
import random
import numpy as np
import torch.nn as nn
import torchsummary
import torch.optim as optim
from torch.utils.data import DataLoader

from helper.FeatureHelper import FeatureHelper
from helper.IOHelper import IOHelper
from sklearn.model_selection import train_test_split
from models.AudioDataSet import AudioDataSet
from models.ConvAudioModel import ConvAudioModel
from tqdm import tqdm


def seed_everything(seed=1016):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


def train(model,
          train_loader,
          test_loader,
          optimizer,
          train_loss=[],
          test_loss=[],
          loss_fn=nn.CrossEntropyLoss(),
          epochs=100,
          change_lr=None):

    for epoch in tqdm(range(0, epochs)):
        model.train()
        batch_loss=[]
        correct = 0

        for i, data in enumerate(train_loader):
            x = data[0]
            y = data[1]
            optimizer.zero_grad()

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()

        train_loss.append(batch_loss)

        print(f'\nEpoch - {epoch} Train-Loss : {np.mean(train_loss[-1])}')
        model.eval()

        batch_loss = []
        trace_y = []
        trace_y_hat = []

        for i, data in enumerate(test_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_y_hat.append(y_hat.cpu().detach().numpy())
            batch_loss.append(loss.item())

        test_loss.append(batch_loss)
        trace_y = np.concatenate(trace_y)
        print(trace_y)
        trace_y_hat = np.concatenate(trace_y_hat)
        accuracy = np.mean(trace_y_hat.argmax(axis=1) == trace_y.argmax(axis=1))
        print(trace_y_hat.argmax(axis=1))
        print(f'\nEpoch - {epoch} Valid-Loss : {np.mean(test_loss[-1])} Valid_Accuracy : {accuracy}')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything()

    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    SOURCE_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\Model\DATA\TRAINING\ORIGINAL\TS01_BRAIN'
    LABEL_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\Model\DATA\TRAINING\LABELED\TL01_BRAIN'

    CLASSES = {
        25: "25_Language",
        26: "26_Ear"
    }

    ioHelper = IOHelper()
    patients = ioHelper.load_file(LABEL_PATH, SOURCE_PATH, CLASSES)

    featureHelper = FeatureHelper()

    start = time.time()
    mfccs = []
    labels = []
    index = 0

    for patient in patients:
        featureFile = './features/' + patient.id + '_MFCC.npy'

        if patient.subType.value == 25 or patient.subType.value == 26:
            if not os.path.exists(featureFile):
                mfcc = featureHelper.extract_all_features(patient.audioFileRoot, patient.id)

                print("Features extracted for %s, disease Code : %d (%d/%d)" % (patient.id, patient.subType.value, index, len(patients)))

                np.save('./features/' + patient.id + '_MFCC.npy', np.array(mfcc))

            else:
                mfcc = np.load(featureFile)
                mfccs.append(mfcc)

        index += 1

        labels.append(patient.subType.value)

    end = time.time()
    print('All Features of patients extracted successfully! ETA : %.5fs' % (end - start))

    X_train, X_test, Y_train, Y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=1016)

    model = ConvAudioModel()

    train_data_set = AudioDataSet(mfcc_list=X_train, labels=Y_train)
    test_data_set = AudioDataSet(mfcc_list=X_test, labels=Y_test)

    train_data_loader = DataLoader(train_data_set, batch_size=20, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_data_set, batch_size=20, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.to(device)

    EPOCHS = 100
    last_best_accuracy = 0.0

    print(mfccs[0].shape)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        model.train()

        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, x)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()
        epoch_loss = running_loss / len(train_data_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

