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
import gc
import json
import librosa
import sys

from skimage.io import imread
from skimage.transform import resize
from glob import glob
from models.ValidationTypeModel import ValidationTypeModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from helper.FeatureHelper import FeatureHelper
from helper.IOHelper import IOHelper
from sklearn.model_selection import train_test_split

from models.LanguageModel import LanguageModel
from torchsummary import summary
from models.AudioDataSet import AudioDataSet


def convert_model_to_mobile():
    model = torch.load('./outputs/T02.pt').to('cpu')
    model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module._save_for_lite_interpreter('./outputs/T02_mobile.pt')

    print('optimize for mobile and model save completed.')


def seed_everything(seed=1006):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_weights(labels, n_classes):
    labels = np.array(labels)
    weight_list = []

    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        count = len(idx)
        weight = 1 / count
        weights = [weight] * count
        weight_list += weights

    return weight_list


def extract_confusion_matrix(test_loader):
    print('Extracting confusion matrix')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model = torch.load('./outputs/T02.pt')
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
    plt.title('Confusion Matrix of T02')
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

    plt.savefig('./confusion_matrix_T02.png', format='png')
    plt.show()


def get_root(type: ValidationTypeModel):
    if type == ValidationTypeModel.WORD:
        return r'\1. Word'

    elif type == ValidationTypeModel.SENTENCE:
        return r'\2. Sentence'

    elif type == ValidationTypeModel.PARAGRAPH:
        return r'\3. Paragraph'

    elif type == ValidationTypeModel.SEMI_FREE_SPEECH:
        return r'\4. Semi_FreeSpeech'

    else:
        raise Exception(f"Unknown Validation Type Model : {type}")


def get_id_and_disease_type(file):
    jsonFile = open(file, 'rt', encoding='UTF8')
    data = json.load(jsonFile)
    cat = data["Disease_info"]["Type"]
    subCat = int(data['Disease_info'][f'Subcategory{int(cat)}'])

    if subCat == 21 or subCat == 24 or subCat == 27:
        return data['File_id'].replace('중복1', '').replace('중복2', ''), subCat

    else:
        return None, None


def get_raw_file_root(disease_type):
    if disease_type == 21:
        return r'\T02\21_Articulation'

    elif disease_type == 24:
        return r'\T02\24_Vocalization'

    elif disease_type == 27:
        return r'\T02\27_Conduction'

    else:
        raise Exception(f"Unknown Disease Type: {disease_type}")


def validation(type: ValidationTypeModel):
    root = r'D:\Projects\DysarthriaChecker\DATA\Additional' + get_root(type)
    parent = r'D:\Projects\DysarthriaChecker\DATA\Original'

    features = []
    labels = glob(root + r'\*\*.json')
    disease_types = []
    id_list = []
    predicted = []

    for label in labels:
        id, disease_type = get_id_and_disease_type(label)

        if id is not None and disease_type is not None:
            id_list.append(id)
            disease_types.append(disease_type)

    print(id_list)
    print(disease_types)

    for idx, d_type in enumerate(disease_types):
        raw_file = parent + get_raw_file_root(d_type) + rf'\{id_list[idx]}'
        y = librosa.load(raw_file, sr=16000)[0]
        S = librosa.feature.melspectrogram(y=y, n_mels=64, n_fft=320, hop_length=160)
        norm_log_S = np.clip((librosa.power_to_db(S, ref=np.max) + 100) / 100, 0, 1)

        figFile = './spectrogram/' + id_list[idx] + '.jpg'

        if not os.path.exists(figFile):
            librosa.display.specshow(norm_log_S, sr=16000, hop_length=160)
            plt.tight_layout()
            plt.savefig(figFile)

        img = imread(figFile)
        features.append(resize(img, (3, 28, 28)))
        del figFile
        gc.collect()

    data_set = AudioDataSet(features, disease_types)
    data_loader = DataLoader(data_set, shuffle=True, drop_last=False)
    actuals = []

    for spec, label in data_loader:
        model = torch.load('./outputs/T02.pt')
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            spec = spec.to(device)
            preds = model(spec.float())
            print(preds)
            predicted.append(torch.argmax(preds))
            actuals.append(label)

    print(actuals)
    print(predicted)
    print(('=' * 10) + type.name + ('=' * 10))



if __name__ == '__main__':
    BATCH_SIZE = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything()
    
    SOURCE_PATH = r'D:\Projects\DysarthriaChecker\DATA\Original\T02'
    LABEL_PATH = r'D:\Projects\DysarthriaChecker\DATA\Labeled\T02'
    
    CLASSES = {
        21: "21_Articulation",
        24: "24_Vocalization",
        27: "27_Conduction"
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
    
        if patient.subType.value == 21 or patient.subType.value == 24 or patient.subType.value == 27:
            if not os.path.exists(featureFile):
                mfcc = featureHelper.extract_all_features(patient.audioFileRoot, patient.id)
    
                print("Features extracted for %s, disease Code : %d (%d/%d)" % (
                patient.id, patient.subType.value, index, len(patients)))
    
                np.save('./features/' + patient.id + '_MFCC.npy', np.array(mfcc))
    
            else:
                mfcc = np.load(featureFile)
                mfccs.append(mfcc)
    
        index += 1
    
        labels.append(patient.subType.value)
    
    end = time.time()
    print('All Features of patients extracted successfully! ETA : %.5fs' % (end - start))
    
    model = LanguageModel().to(device)
    
    imgs = []
    
    for (i, mfcc) in enumerate(mfccs):
        figFile = './spectrogram/' + patients[i].id + '.jpg'
    
        if not os.path.exists(figFile):
            librosa.display.specshow(mfcc, sr=16000, hop_length=160)
            plt.tight_layout()
    
            plt.savefig(figFile)
            print("Spectrogram for %s saved (%d/%d)" % (patients[i].id, i + 1, len(mfccs)))
            plt.close()
            del figFile
            gc.collect()
    
        del mfcc
        img = imread(figFile)
        imgs.append(resize(img, (3, 28, 28)))
        del img
    
    print("All Spectrogram of patients extracted successfully!")
    
    imgs = np.array(imgs)
    
    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1006)
    train_data_set = AudioDataSet(X_train, Y_train)
    train_data_loader = DataLoader(train_data_set, batch_size=20, shuffle=True, drop_last=False)
    
    test_data_set = AudioDataSet(X_test, Y_test)
    test_data_loader = DataLoader(test_data_set, shuffle=False, drop_last=False)
    
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
                predicted_labels_batch = torch.argmax(targets, dim=1)
    
                total += label.size(0)
                correct += (predicted_labels_batch == torch.argmax(label, dim=1)).sum().item()
    
        accuracy = correct / total
    
        if accuracy > last_best_acc:
            last_best_acc = accuracy
            torch.save(model, './outputs/T02.pt')
            print('best accuracy model saved ', accuracy)
    
        print(
            f"Epoch {epoch + 1} Loss: {running_loss / (len(train_data_set) / BATCH_SIZE)} Accuracy: {accuracy:.3f} (best: {last_best_acc:.3f})")
    
    print(f'Train Finished. last best accuracy : {last_best_acc:.3f}')
    extract_confusion_matrix(test_data_loader)
    
    convert_model_to_mobile()
