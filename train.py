import torch
import torch.nn as nn
import torch.optim as optim
from networks import MLP
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import deque
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

loss_fn = nn.MSELoss()

# ********************************************** Training a neural network to imply volatility *********************************************

implied_vol_config = dict(
    input_dim=3,
    output_dim=1,
    nb_neurons=400,
    depth=4,
    activation=nn.SiLU(),
    positive=True,
    lr=0.005,
    batch_size=1024,
    epochs=5000,
)

model = MLP(
    implied_vol_config["input_dim"],
    implied_vol_config["output_dim"],
    implied_vol_config["nb_neurons"],
    implied_vol_config["depth"],
    activation=implied_vol_config["activation"],
).to(device)

optimizer = optim.SGD(model.parameters(), lr=implied_vol_config["lr"], momentum=0.9)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, step_size_down=10
)

X = np.load("X.npy")
Y = np.load("Y.npy")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)


print("########### Loading Data ############")
dtype = torch.float
dataset_train = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).type(dtype), torch.from_numpy(Y_train).type(dtype)
)
dataset_test = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test).type(dtype), torch.from_numpy(Y_test).type(dtype)
)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1024, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1024, shuffle=False)


print("########### Data succesfully loaded ############")


def train(model, optimizer, epochs, desc):
    training_loss = deque()
    validation_loss = deque()
    with tqdm.tqdm(total=epochs, desc=desc, position=0, leave=True) as progress_bar:
        best_validation_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in loader_train:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss = train_loss / len(loader_train)
            training_loss.append(train_loss)
            if epoch % 5 == 0:
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for inputs, targets in loader_test:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        valid_loss += loss.item()
                valid_loss = valid_loss / len(loader_test)
                validation_loss.append(valid_loss)
                """if valid_loss < best_validation_loss:
                    print("Checkpoint")
                    torch.save(model.state_dict(), "checkpoint.pth")
                    best_validation_loss = valid_loss"""

                logs = f"Epoch: {epoch}, lr = {scheduler.get_last_lr()}, training_loss = {train_loss}, validation_loss = {valid_loss}"
            else:
                logs = f"Epoch: {epoch}, lr = {scheduler.get_last_lr()}, training_loss = {train_loss}"
            scheduler.step()
            print(logs)
            # progress_bar.update()
            # progress_bar.set_description(desc=logs
            if epoch % 500 == 0 and epoch > 0:
                np.save("training_loss_iv", training_loss)
                np.save("validation_loss_iv", validation_loss)

    return training_loss, validation_loss


# ******************************************************************************************************************************************


if __name__ == "__main__":
    print(device)
    # ********************** Implied volatility training ****************************
    model.load_state_dict(torch.load("checkpoint.pth"))
    model = model.to(device)
    training_loss, validation_loss = train(
        model, optimizer, implied_vol_config["epochs"], "Training Implied Vol"
    )
    np.save("training_loss_iv", training_loss)
    np.save("validation_loss_iv", validation_loss)
    torch.save(model.state_dict(), "checkpoint2.pth")
    # ********************************************************************************
