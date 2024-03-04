import time
import librosa
import torch
import os
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc
import json

from skimage.io import imread
from skimage.transform import resize
from glob import glob
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

from helper.FeatureHelper import FeatureHelper
from helper.IOHelper import IOHelper
from sklearn.model_selection import train_test_split

from models.AudioDataSet import AudioDataSet
from models.BrainModel import BrainModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models.ValidationTypeModel import ValidationTypeModel


def convert_model_to_mobile():
    model = torch.load('./outputs/model.pt').to('cpu')
    model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module._save_for_lite_interpreter('./outputs/T01_mobile.pt')

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

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix of T01')
    plt.colorbar()

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(confusion[i, j]), horizontalalignment='center', verticalalignment='center',
                     color='white')

    class_names = ['0', '1']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig('./confusion_matrix_T01.png', format='png')
    plt.show()


def seed_everything(seed=1006):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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

    if subCat == 25 or subCat == 26:
        return data['File_id'].replace('중복1', '').replace('중복2', ''), subCat

    else:
        return None, None


def get_raw_file_root(disease_type):
    if disease_type == 25:
        return r'\T01\25_Language'

    elif disease_type == 26:
        return r'\T01\26_Ear'

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
        model = torch.load('./outputs/T01.pt')
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

    weights = [1.25, 0.75]
    
    seed_everything()
    
    SOURCE_PATH = r'D:\Projects\DysarthriaChecker\DATA\Original\T01'
    # LABEL_PATH = r'C:\Users\USER\Desktop\2023\DysarthriaChecker\Model\DATA\TRAINING\LABELED\TL03_LARYNX'
    LABEL_PATH = r'D:\Projects\DysarthriaChecker\DATA\Labeled\T01'
    
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
    imgs = []
    
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
    
    imgs = []
    
    for (i, mfcc) in (enumerate(mfccs)):
        figFile = './spectrogram/' + patients[i].id + '.jpg'
    
        if not os.path.exists(figFile):
            librosa.display.specshow(mfcc, sr=16000, hop_length=160)
            plt.tight_layout()
    
            plt.savefig(figFile)
            print("Spectrogram for %s saved (%d/%d)" % (patients[i].id, i + 1, len(mfccs)))
    
        img = imread(figFile)
        imgs.append(resize(img, (3, 28, 28)))
    
    print("All Spectrogram of patients extracted successfully! imgs.size : %d, labels.size : %d" % (
    len(imgs), len(labels)))
    
    imgs = np.array(imgs)
    
    X_train, X_test, Y_train, Y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1006)
    train_data_set = AudioDataSet(X_train, Y_train)
    train_data_loader = DataLoader(train_data_set, batch_size=20, shuffle=True, drop_last=False)
    
    test_data_set = AudioDataSet(X_test, Y_test)
    test_data_loader = DataLoader(test_data_set, shuffle=False, drop_last=False)
    
    model = BrainModel()
    model = model.to(device)
    weights = torch.Tensor(weights)
    criterion = nn.BCELoss(weight=weights.to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    EPOCHS = 100
    last_best_acc = 0.0
    last_best_precision = 0.0
    last_best_recall = 0.0
    last_best_F1 = 0.0
    
    class_sample_counts_train = [0] * 2
    
    for _, labels in train_data_loader:
        for label in labels:
            class_sample_counts_train[label.item()] += 1
    
    class_sample_counts_test = [0] * 2
    
    for _, labels in test_data_loader:
        for label in labels:
            class_sample_counts_test[label.item()] += 1
    
    print(f'Train started\ncounts of train 0: {class_sample_counts_train[0]}, 1: {class_sample_counts_train[1]}\ncounts of test 0: {class_sample_counts_test[0]}, 1: {class_sample_counts_test[1]}')
    
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
    
        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            y = F.one_hot(y % 3, num_classes=2)
    
            optimizer.zero_grad()
            outputs = model(x.float())
    
            loss = criterion(outputs.to(torch.float32), y.to(torch.float32))
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []
    
        model.eval()
    
        with torch.no_grad():
            for spec, label in test_data_loader:
                spec = spec.to(device)
                label = label.to(device)
                label = F.one_hot(label % 3, num_classes=2)
    
                targets = model(spec.float())
                predicted_labels_batch = torch.argmax(targets, dim=1)
    
                true_labels.extend(label[:, 1].cpu().numpy())
                predicted_labels.extend(predicted_labels_batch.cpu().numpy())
    
                total += label.size(0)
                correct += (predicted_labels_batch == torch.argmax(label, dim=1)).sum().item()
    
        accuracy = correct / total
        precison = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
    
        if accuracy > last_best_acc:
            last_best_acc = accuracy
            torch.save(model, './outputs/model.pt')
            print('best accuracy model saved ', accuracy)
    
        if precison > last_best_precision:
            last_best_precision = precison
    
        if recall > last_best_recall:
            last_best_recall = recall
    
        if f1 > last_best_F1:
            last_best_F1 = f1
    
        print(
            f"Epoch {epoch + 1} Loss: {running_loss / (len(train_data_set) / BATCH_SIZE)} Accuracy: {accuracy:.3f} (best: {last_best_acc:.3f})\nPrecision: {precison:.3f} (best: {last_best_precision:.3f}) Recall: {recall:.3f} (best: {last_best_recall:.3f}) F1: {f1:.3f} (best: {last_best_F1:.3f})")
    
    print(f'Train Finished. last best accuracy : {last_best_acc:.3f}')
    extract_confusion_matrix(test_data_loader)
    
    convert_model_to_mobile()
