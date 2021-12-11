from comet_ml import Experiment
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import CometLogger
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from models.utils import load_orig_data, load_processed_data, save_preds
load_dotenv()
from datetime import datetime
SAVE_PREDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "preds")

# comet_logger = CometLogger(
#     api_key=os.environ.get("COMET_API"),
# 	project_name="data-challenge-2",
# 	workspace="ift6390-datachallenge-2",
# )

class TrainData(Dataset):

	def __init__(self, X_data, y_data):
		self.X_data = X_data
		self.y_data = y_data

	def __getitem__(self, index):
		return self.X_data[index], self.y_data[index]

	def __len__(self):
		return len(self.X_data)


class ValData(Dataset):

	def __init__(self, X_data, y_data):
		self.X_data = X_data
		self.y_data = y_data

	def __getitem__(self, index):
		return self.X_data[index], self.y_data[index]

	def __len__(self):
		return len(self.X_data)


class TestData(Dataset):

	def __init__(self, X_data):
		self.X_data = X_data

	def __getitem__(self, index):
		return self.X_data[index]

	def __len__(self):
		return len(self.X_data)


class NN_Model(pl.LightningModule):
	def __init__(self, NUM_INPUT_FEATURES=216):
		super().__init__()
		self.FC = nn.Sequential(
			nn.Linear(NUM_INPUT_FEATURES, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.FC(x)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x - x.view(x.size(0), -1)
		y_hat = self.FC(x)
		y_hat = y_hat.squeeze()
		criterion = torch.nn.BCELoss()
		loss = criterion(y_hat, y)
		f1_score = torchmetrics.functional.f1(y_hat, y.long())
		self.log('train_loss', loss)
		self.log('train_f1_score', f1_score)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x - x.view(x.size(0), -1)
		y_hat = self.FC(x)
		y_hat = y_hat.squeeze()
		criterion = torch.nn.BCELoss()
		loss = criterion(y_hat, y)
		f1_score = torchmetrics.functional.f1(y_hat, y.long())
		self.log('val_loss', loss)
		self.log('val_f1_score', f1_score)


def get_dataloaders(batch_size, processed_name=None):
	if processed_name is None:
		X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
	else:
		X_train, X_val, y_train, y_val, X_test = load_processed_data(version=processed_name, split_val=True)

	train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(y_train.to_numpy()))
	val_data = TrainData(torch.FloatTensor(X_val.to_numpy()), torch.FloatTensor(y_val.to_numpy()))
	test_data = torch.FloatTensor(X_test.to_numpy())

	train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

	return train_loader, val_loader, test_data


def run_experiment(batch_size=64):
	train_loader, val_loader, test_data = get_dataloaders(batch_size=batch_size)
	model = NN_Model()
	early_stop_callback = EarlyStopping(monitor="val_loss")
	trainer = pl.Trainer(max_epochs=25, callbacks=[early_stop_callback])
	trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
	preds = model(test_data)
	preds = torch.as_tensor((preds - 0.5) >= 0, dtype=torch.int32)

	save_preds(preds,'Neural_Network',trainer.logged_metrics,{})


def main():
	run_experiment()


if __name__ == "__main__":
	main()
