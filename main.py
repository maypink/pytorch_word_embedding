from torch.utils.data import DataLoader
from CBOWNet import CBOWNet
from Word2VecDataset import Word2VecDataset
from training import training

dataset = Word2VecDataset()
model = CBOWNet(len(dataset.vocab), 50)
data_loader = DataLoader(dataset, batch_size=500)
training(model, data_loader)