import torch
import torch.nn as nn
import pytorch_lightning as pl


class SiameseTriplet(pl.LightningModule):
    def __init__(self, embedding_dim=120, margin=1.0, learning_rate=0.001):
        super(SiameseTriplet, self).__init__()

        self.validation_step_outputs = []
        self.train_step_outputs = []

        # Fully connected layers for prediction
        self.fc_query = nn.Linear(embedding_dim, 256)
        self.fc_doc = nn.Linear(embedding_dim, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

        self.margin = margin
        self.learning_rate = learning_rate
        self.criterion = nn.TripletMarginLoss(margin=margin)

    def forward_one(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.forward_one(self.fc_query(anchor))
        positive_embedding = self.forward_one(self.fc_doc(positive))
        negative_embedding = self.forward_one(self.fc_doc(negative))
        
        return anchor_embedding, positive_embedding, negative_embedding

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        out_anchor, out_positive, out_negative = self(anchor, positive, negative)
        loss = self.criterion(out_anchor, out_positive, out_negative)
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        out_anchor, out_positive, out_negative = self(anchor, positive, negative)
        loss = self.criterion(out_anchor, out_positive, out_negative)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()
        if not len(self.validation_step_outputs) == 0:
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            self.log("validation_epoch_average", epoch_average)
            print("val_loss_avg: ", epoch_average)
            self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SiameseNetworkPL(pl.LightningModule):
    def __init__(self, input_size, output_size, learning_rate=1e-3, margin=1.0, arch_type=None):
        super(SiameseNetworkPL, self).__init__()

        self.validation_step_outputs = []
        self.train_step_outputs = []

        # Network architecture
        if arch_type == 'linear':
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            )
        elif arch_type == 'conv':
            self.network = nn.Sequential(
                # Reshape layer to add a channel dimension (N, C, L)
                nn.Unflatten(1, (1, input_size)),

                # 1D Convolution layers
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Flatten(),   # Flatten the output for the linear layers

                # Linear layers + dropout
                nn.Linear(64 * (input_size // 4), 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(128, output_size)
            )
        elif arch_type == 'lstm':
            self.network = nn.Sequential(
            # LSTM layer
            nn.LSTM(
                input_size=input_size,  # Assuming each time step of the sequence is of 'input_size' dimension
                hidden_size=128,        # Size of the hidden state
                num_layers=2,           # Number of LSTM layers
                batch_first=True,       # Input and output tensors are provided as (batch, seq, feature)
                dropout=0.2             # Dropout for regularization
            ),
            # Flatten the output for the fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )


        self.margin = margin
        self.learning_rate = learning_rate
        self.criterion = nn.TripletMarginLoss(margin=margin)

    def forward(self, x):
        # Forward pass for one input
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # Training step
        anchor, positive, negative = batch
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        # Calculate triplet loss
        loss = self.criterion(anchor_output, positive_output, negative_output)
        # Append loss to list for epoch average
        self.train_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step, similar to training_step
        anchor, positive, negative = batch
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)

        # Calculate loss
        loss = self.criterion(anchor_output, positive_output, negative_output)
        # Append loss to list for epoch average
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()
        if not len(self.validation_step_outputs) == 0:
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            self.log("validation_epoch_average", epoch_average)
            print("val_loss_avg: ", epoch_average)
            self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    