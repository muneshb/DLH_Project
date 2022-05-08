import os
import sys

scriptpath = "C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\"
sys.path.append(os.path.abspath(scriptpath))

import pickle
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from data_load import PatientSimilarity
from patient_networks import EmbeddingNet, PatientNet
from nn_losses import TripletLoss
from nn_trainer import fit
cuda = torch.cuda.is_available()


with open('C:\\Users\\14088\\Documents\\Books\\CS598 - DLH\\paper32\\code\\tensor_data.pickle', 'rb') as handle:
    patient_dataset = pickle.load(handle)

n_classes = 3

patient_train_dataset = PatientSimilarity(patient_dataset, train=True)
patient_test_dataset = PatientSimilarity(patient_dataset, train = False)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
patient_train_loader = torch.utils.data.DataLoader(patient_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
patient_test_loader = torch.utils.data.DataLoader(patient_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)



margin = 1.
embedding_net = EmbeddingNet()
model = PatientNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(patient_train_loader, patient_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

def get_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data_arr, target in dataloader:
            if cuda:
                data_arr = data_arr.cuda()
            embeddings[k:k+len(data_arr)] = model.get_embedding(data_arr).data.cpu().numpy()
            labels[k:k+len(data_arr)] = target.numpy()
            k += len(data_arr)
    return embeddings, labels

#train_data_embeddings, train_data_labels = get_embeddings(patient_train_loader, model)
#test_data_embeddings, test_data_labels = get_embeddings(patient_test_loader, model)


