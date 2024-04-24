import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import Transformer
import math
import random


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BaseTransformer(pl.LightningModule):
    def __init__(self, ntoken=None, d_model=120, nhead=4, nhid=120, nlayers=3, dropout=0.2, teacher_forcing_prob=1,hparams=None):
        super(BaseTransformer, self).__init__()
        self.save_hyperparameters(hparams)

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.train_accuracy_outputs = []
        self.validation_accuracy_outputs = []

        self.teacher_forcing_prob = teacher_forcing_prob


        self.model = Transformer(d_model=d_model, nhead=nhead,
                                 num_encoder_layers=nlayers, num_decoder_layers=nlayers,
                                 dim_feedforward=nhid, dropout=dropout)
        self.src_tok_emb = nn.Embedding(ntoken, d_model, padding_idx=0)
        self.tgt_tok_emb = nn.Embedding(13, d_model, padding_idx=11)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.relu = nn.ReLU()

        # Update the output layer dimensions
        self.output_layer = nn.Linear(d_model, 13)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_masks(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).type(torch.bool).to(self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == 0).transpose(0, 1).type(torch.bool)  # src == 0 is for padding_idx
        tgt_padding_mask = (tgt == 11).transpose(0, 1).type(torch.bool)  # tgt == 11 is for padding_idx
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt)

        src = self.src_tok_emb(src)
        tgt = self.tgt_tok_emb(tgt)

        src = self.positional_encoding(src).to(self.device)
        tgt = self.positional_encoding(tgt).to(self.device)

        output = self.model(src, tgt, src_mask, tgt_mask,
                            None, src_padding_mask, tgt_padding_mask, src_padding_mask)

        return self.output_layer(output)


    def training_step(self, batch, batch_idx):
        src, tgt = batch

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)


        # Extract the elements that don't match the specified value
        #tgt_input = tgt[:-1, :]
        #tgt_output = tgt[1:, :]
        #print(self.teacher_forcing_prob)

        use_teacher_forcing = random.random() < self.teacher_forcing_prob

        if use_teacher_forcing:
            # Use the ground truth for the entire sequence
            tgt_input = tgt[:-1, :]
            output = self(src, tgt_input)

        else:  #AUTOREGRESSIVE SETUP
            # Generate the sequence token by token using the model's own predictions
            # Initialize the output tensor
            output = torch.zeros(tgt.size(0) - 1, src.size(1), 13, device=src.device)

            tgt_input = tgt[:1, :]  # Start with the first token

            for t in range(1, tgt.size(0)):
                output_t = self(src, tgt_input)

                # Extract the prediction for the last token and store it
                last_token_output = output_t[-1, :, :].unsqueeze(0)
                output[t - 1] = last_token_output.squeeze(0)

                # Update tgt_input for the next step
                next_token = torch.argmax(last_token_output, dim=-1)
                tgt_input = torch.cat((tgt_input, next_token), dim=0) #we look at the sequence generated up to the current token


        tgt_output = tgt[1:, :]
        #print(use_teacher_forcing, output.shape, tgt_output.shape)

        #output = self(src, tgt_input)  #shape: (docid_len, batch_size, docid_vocab_size)

        loss = self.calculate_loss(output, tgt_output)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_step_outputs.append(loss)

        # Compute accuracy
        # Take the argmax of the output to get the most likely token
        predictions = torch.argmax(output, dim=-1)
        # Compute accuracy with masking for padding
        non_padding_mask = (tgt_output != 11)
        correct_count = ((predictions == tgt_output) & non_padding_mask).sum().item()
        total_count = non_padding_mask.sum().item()
        # Avoid division by zero
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        accuracy_tensor = torch.tensor(accuracy)
        # Log training loss and accuracy
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.train_accuracy_outputs.append(accuracy_tensor)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Autoregressive setup for validation
        # Start with the first token
        output = torch.zeros(tgt.size(0) - 1, src.size(1), 13, device=src.device)

        tgt_input = tgt[:1, :]  # Start with the first token

        for t in range(1, tgt.size(0)):
            output_t = self(src, tgt_input)

            # Extract the prediction for the last token and store it
            last_token_output = output_t[-1, :, :].unsqueeze(0)
            output[t - 1] = last_token_output.squeeze(0)

            # Update tgt_input for the next step
            next_token = torch.argmax(last_token_output, dim=-1)
            tgt_input = torch.cat((tgt_input, next_token), dim=0) #we look at the sequence generated up to the current token

        tgt_output = tgt[1:, :]

        loss = self.calculate_loss(output, tgt_output)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)

        # Compute accuracy
        predictions = torch.argmax(output, dim=-1)
        non_padding_mask = (tgt_output != 11)  # Assuming 11 is your padding token
        correct_count = ((predictions == tgt_output) & non_padding_mask).sum().item()
        total_count = non_padding_mask.sum().item()
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        accuracy_tensor = torch.tensor(accuracy)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.validation_accuracy_outputs.append(accuracy_tensor)

        return {"val_loss": loss}


    def on_validation_epoch_end(self):

        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            #print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()

            epoch_train_acc = torch.stack(self.train_accuracy_outputs).mean()
            self.log("train_accuracy", epoch_train_acc)
            #print("train_acc_avg: ", epoch_train_acc)
            self.train_accuracy_outputs.clear()

        if not len(self.validation_step_outputs) == 0:
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            self.log("validation_epoch_average", epoch_average)
            #print("val_loss_avg: ", epoch_average)
            self.validation_step_outputs.clear()

            epoch_val_acc = torch.stack(self.validation_accuracy_outputs).mean()
            self.log("validation_accuracy", epoch_val_acc)
            #print("val_acc_avg: ", epoch_val_acc)
            self.validation_accuracy_outputs.clear()
        #at each validation epoch, we reduce the teacher forcing presence     
        self.teacher_forcing_prob -= 0.02

    def calculate_loss(self, output, target):
        criterion = nn.CrossEntropyLoss(ignore_index=11)  # Use docid padding token
        output = output.view(-1, output.shape[-1])
        target = target.contiguous().view(-1)
        return criterion(output, target)

    def configure_optimizers(self):
        # Define and return optimizer. Example: Adam
        return torch.optim.Adam(self.parameters(), lr=0.0005)