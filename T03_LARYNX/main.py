import time
import librosa.display
import torch
import os
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc

from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from helper.FeatureHelper import FeatureHelper
from helper.IOHelper import IOHelper
from sklearn.model_selection import train_test_split

from models.LarynxModel import LarynxModel
from torchsummary import summary
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
    optimized_traced_model.save('./outputs/T03_mobile.pt')

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

    SOURCE_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\Model\DATA\TRAINING\ORIGINAL\TS03_LARYNX'
    # LABEL_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\Model\DATA\TRAINING\LABELED\TL03_LARYNX'
    LABEL_PATH = r'D:\DysarthriaChecker_Original\DATA\TL03'

    CLASSES = {
        31: "31_Functional",
        32: "32_Larynx",
        33: "33_Oral"
    }

    ioHelper = IOHelper()
    patients = ioHelper.load_file(LABEL_PATH, SOURCE_PATH, CLASSES)

    featureHelper = FeatureHelper()

    start = time.time()
    mfccs = []
    labels = []
    index = 0
    imgs = []

    for patient in patients:
        # featureFile = './features/' + patient.id + '_MFCC.npy'

        if patient.subType.value == 31 or patient.subType.value == 32 or patient.subType.value == 33:
            figFile = 'D:/DysarthriaChecker_Original/DATA/Features/spectrogram/T03_LARYNX/' + patient.id + '.jpg'
            img = imread(figFile)
            imgs.append(resize(img, (3, 28, 28)))

            # if not os.path.exists(featureFile):
            #     mfcc = featureHelper.extract_all_features(patient.audioFileRoot, patient.id)
            #
            #     print("Features extracted for %s, disease Code : %d (%d/%d)" % (patient.id, patient.subType.value, index, len(patients)))
            #
            #     np.save('./features/' + patient.id + '_MFCC.npy', np.array(mfcc))
            #
            # else:
            #     mfcc = np.load(featureFile)
            #     mfccs.append(mfcc)

            labels.append(patient.subType.value)

        # index += 1


    end = time.time()
    print('All Features of patients extracted successfully! ETA : %.5fs' % (end - start))

    # imgs = []
    #
    # for (i, patient) in enumerate(patients):
    #     figFile = './spectrogram/' + patients[i].id + '.jpg'
    #     img = imread(figFile)
    #     imgs.append(resize(img, (3, 28, 28)))
    #
    # for (i, mfcc) in reversed(list((enumerate(mfccs)))):
    #     figFile = './spectrogram/' + patients[i].id + '.jpg'
    #
    #     if not os.path.exists(figFile):
    #         librosa.display.specshow(mfcc, sr=16000, hop_length=160)
    #         plt.tight_layout()
    #
    #         plt.savefig(figFile)
    #         print("Spectrogram for %s saved (%d/%d)" % (patients[i].id, i+1, len(mfccs)))
    #         plt.close()
    #         del figFile
    #         gc.collect()
    #
    #     img = imread(figFile)
    #     imgs.append(resize(img, (3, 28, 28)))

    print("All Spectrogram of patients extracted successfully! imgs.size : %d, labels.size : %d" % (len(imgs), len(labels)))

    imgs = np.array(imgs)

    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1016)
    train_data_set = AudioDataSet(X_train, Y_train)
    train_data_loader = DataLoader(train_data_set, batch_size=20, shuffle=True, drop_last=False)

    test_data_set = AudioDataSet(X_test, Y_test)
    test_data_loader = DataLoader(test_data_set, shuffle=False, drop_last=False)

    model = LarynxModel()
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
