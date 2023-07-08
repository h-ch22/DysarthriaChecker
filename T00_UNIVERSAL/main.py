import glob
import time
import librosa.display
import torch
import os
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.model_selection import train_test_split

from models.BrainModel import BrainModel
from skimage.io import imread
from skimage.transform import resize
from models.AudioDataSet import AudioDataSet


def convert_model_to_mobile():
    model = torch.load('./outputs/model.pt')
    model = model.to(device)
    model.eval()
    example = torch.rand(1, 3, 28, 28)
    example = example.to(device)
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model.save('./outputs/T00_mobile.pt')

    print('optimize for mobile and model save completed.')


def seed_everything(seed=1016):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    BATCH_SIZE = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything()

    T01_SOURCE_PATH = r'D:\DysarthriaChecker_Original\DATA\Features\spectrogram\T01_BRAIN'
    T02_SOURCE_PATH = r'D:\DysarthriaChecker_Original\DATA\Features\spectrogram\T02_LANGUAGE'
    T03_SOURCE_PATH = r'D:\DysarthriaChecker_Original\DATA\Features\spectrogram\T03_LARYNX'

    labels = []
    imgs = []

    T01_files = glob.glob(T01_SOURCE_PATH + r'\*.jpg')
    T02_files = glob.glob(T02_SOURCE_PATH + r'\*.jpg')
    T03_files = glob.glob(T03_SOURCE_PATH + r'\*.jpg')

    for fig_img in T01_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        labels.append(0)

    for fig_img in T02_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        labels.append(1)

    for fig_img in T03_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        labels.append(2)

    print("All Spectrogram of patients extracted successfully!")

    imgs = np.array(imgs)

    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1016)
    train_data_set = AudioDataSet(X_train, Y_train)
    train_data_loader = DataLoader(train_data_set, batch_size=20, shuffle=True, drop_last=False)

    test_data_set = AudioDataSet(X_test, Y_test)
    test_data_loader = DataLoader(test_data_set, shuffle=False, drop_last=False)

    model = BrainModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    EPOCHS = 4000
    last_best_acc = 0.0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()

        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            y = F.one_hot(y % 3, num_classes=3)

            optimizer.zero_grad()
            outputs = model(x.float())

            loss = criterion(outputs.to(torch.float32), y.to(torch.float32))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():
            for spec, label in test_data_loader:
                spec = spec.to(device)
                label = label.to(device)
                label = F.one_hot(label % 3, num_classes=3)

                targets = model(spec.float())
                predicted_labels = torch.argmax(targets, dim=1)

                total += label.size(0)
                correct += (predicted_labels == torch.argmax(label, dim=1)).sum().item()

        accuracy = correct / total

        if accuracy > last_best_acc:
            last_best_acc = accuracy
            torch.save(model, './outputs/model.pt')
            print('best accuracy model saved ', accuracy)

        print(f"Epoch {epoch + 1} Loss: {running_loss / (len(train_data_set) / BATCH_SIZE)} Accuracy: {accuracy}")

    print(f'Train Finished. last best accuracy : {last_best_acc:.3f}')

    convert_model_to_mobile()

