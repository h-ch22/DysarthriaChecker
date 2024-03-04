import gc
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
import seaborn as sns

from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.model_selection import train_test_split

from models.BrainModel import BrainModel
from skimage.io import imread
from skimage.transform import resize
from models.AudioDataSet import AudioDataSet
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def convert_model_to_mobile():
    model = torch.load('./outputs/model.pt').to('cpu')
    model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module._save_for_lite_interpreter('./outputs/T00_mobile.pt')

    print('optimize for mobile and model save completed.')


def extract_confusion_matrix(test_loader):
    print('Extracting confusion matrix')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model = torch.load('./outputs/model.pt')
    test_model = test_model.to(device)

    true = []
    predicted = []

    test_model.eval()

    with torch.no_grad():
        for spec, label in test_loader:
            spec = spec.to(device)
            label = label.to(device)

            targets = test_model(spec.float())
            predicted_labels_batch = torch.argmax(targets, dim=1)

            true.extend(label.cpu().numpy())
            predicted.extend(predicted_labels_batch.cpu().numpy())

    confusion = confusion_matrix(true, predicted)
    ax = plt.subplot()
    sns.heatmap(confusion, annot=True, fmt='g', ax=ax)
    plt.figure(figsize=(8, 6))
    plt.matshow(confusion)
    plt.title('Confusion Matrix of T00')
    plt.colorbar()

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(confusion[i, j]), horizontalalignment='center', verticalalignment='center',
                     color='white')

    class_names = ['0', '1', '2']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig('./confusion_matrix_T00.png', format='png')
    plt.show()


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

    T01_SOURCE_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\src\ML\T01_BRAIN\spectrogram'
    T02_SOURCE_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\src\ML\T02_LANGUAGE\spectrogram'
    T03_SOURCE_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\src\ML\T03_LARYNX\spectrogram'

    labels = []
    imgs = []

    T01_files = glob.glob(T01_SOURCE_PATH + r'\*.jpg')
    T02_files = glob.glob(T02_SOURCE_PATH + r'\*.jpg')
    T03_files = glob.glob(T03_SOURCE_PATH + r'\*.jpg')

    for fig_img in T01_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        print('Spectrogram loaded: %s' % fig_img)

        del img
        gc.collect()
        labels.append(0)

    for fig_img in T02_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        print('Spectrogram loaded: %s' % fig_img)

        del img
        gc.collect()
        labels.append(1)

    for fig_img in T03_files:
        img = imread(fig_img)
        imgs.append(resize(img, (3, 28, 28)))
        print('Spectrogram loaded: %s' % fig_img)

        del img
        gc.collect()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 100
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
                predicted = torch.argmax(targets, dim=1)

                total += label.size(0)
                correct += (predicted == torch.argmax(label, dim=1)).sum().item()

        accuracy = correct / total

        if accuracy > last_best_acc:
            last_best_acc = accuracy
            torch.save(model, './outputs/model.pt')
            print('best accuracy model saved ', accuracy)

        print(
            f"Epoch {epoch + 1} Loss: {running_loss / (len(train_data_set) / BATCH_SIZE)} Accuracy: {accuracy:.3f} (best: {last_best_acc:.3f})")

    print(f'Train Finished. last best accuracy : {last_best_acc:.3f}')

    convert_model_to_mobile()

    extract_confusion_matrix(test_data_loader)
