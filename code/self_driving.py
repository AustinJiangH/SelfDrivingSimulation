'''


'''



import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


from dataset import read_driving_records, DrivingDataset, get_driving_data_loaders
from model import SelfDrivingModel
from train import train_model



# hyper parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
NUM_WORKERS = 8
SHUFFLE = True

NUM_EPOCHS = 50
START_EPOCH = 0
RESUME = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# working directory is code/
path = '../data/online1'
train_records, valid_records, test_records = read_driving_records(path)
train_dataset, valid_dataset, test_dataset = DrivingDataset(path, train_records), DrivingDataset(path, valid_records), DrivingDataset(path, test_records)
train_loader, valid_loader, test_loader = get_driving_data_loaders(32, train_dataset, valid_dataset, test_dataset)



# create model
model = SelfDrivingModel()

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

# train the model 
train_model(model, NUM_EPOCHS, train_loader, valid_loader, test_loader, optimizer, DEVICE, scheduler=scheduler)
