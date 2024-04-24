import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        seq_len, batch_size, feature_dim = x.size()

        position = torch.arange(0, seq_len).unsqueeze(1).float().to(x.device)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim)).to(x.device)
        pe = torch.zeros(seq_len, 1, feature_dim).to(x.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term[:feature_dim//2])
        pe[:, 0, 1::2] = torch.cos(position * div_term[:feature_dim//2])
        pe = pe.unsqueeze(0)

        # Add positional encoding and apply layer normalization
        x_with_pe = F.layer_norm(x + pe, x.size()[1:]).squeeze(0)

        return x_with_pe

class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self, token_vocab_size, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.train_accuracy_outputs = []
        self.validation_accuracy_outputs = []

        # Embedding layer
        self.embedding = nn.Embedding(token_vocab_size, d_model, padding_idx=0)
        nn.init.orthogonal_(self.embedding.weight)

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer layers with layer normalization and dropout
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=False,
            dropout=dropout
        )

        # Linear layer with layer normalization
        self.fc = nn.Linear(d_model, token_vocab_size)
        self.layer_norm = nn.LayerNorm(token_vocab_size)
    
    def forward(self, input_ids, target_ids):
        # Embedding
        input_embedding = self.embedding(input_ids)
        target_embedding = self.embedding(target_ids)

        # Add positional encoding
        input_embedding = self.positional_encoding(input_embedding)
        target_embedding = self.positional_encoding(target_embedding)

        # Create padding masks
        src_padding_mask = (input_ids == 0).transpose(0, 1).to(input_ids.device)
        tgt_padding_mask = (target_ids == 0).transpose(0, 1).to(target_ids.device)

        # Create the causal mask for the decoder
        tgt_seq_len = target_embedding.size(0)
        tgt_causal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).to(torch.bool).to(target_embedding.device)

        # Transformer with masks
        output_transformer = self.transformer(input_embedding, target_embedding,
                                              src_key_padding_mask=src_padding_mask,
                                              tgt_key_padding_mask=tgt_padding_mask,
                                              memory_key_padding_mask=src_padding_mask,
                                              tgt_mask=tgt_causal_mask)

        # Reshape the output from the transformer to be compatible with the linear layer
        output_flat = output_transformer.view(-1, output_transformer.size(-1))

        # Apply linear layer, layer normalization, and softmax activation
        output = self.layer_norm(F.softmax(self.fc(output_flat), dim=-1))

        # Reshape the output back to (seq_len, batch_size, vocab_size)
        output = output.view(tgt_seq_len, -1, output.size(-1))

        return output

    def training_step(self, batch, batch_idx):
        input_ids, target_ids, _ = batch

        # Ensure that input_ids and target_ids have the correct dimensions (sequence length, batch size)
        input_ids = input_ids.permute(1, 0)
        target_ids = target_ids.permute(1, 0)

        # Pass the sequence-first tensors to the transformer
        output = self(input_ids, target_ids)

        '''
        # Initialize decoder input with the padding token (assuming padding_token_id is defined)
        decoder_input = torch.tensor([[0]] * target_ids.size(1)).to(target_ids.device)

        # Initialize the loss
        loss = 0.0

        # Initialize correct count and total count
        correct_count = 0
        total_count = 0

        # Loop over each time step in the target sequence
        for t in range(target_ids.size(0)):
            # Use teacher forcing: replace the decoder input with the true target up to the current time step
            true_target_sequence = target_ids[:t + 1]
            padding_size = target_ids.size(0) - true_target_sequence.size(0)

            # Create a padding tensor with the correct batch size
            padding_vector = torch.tensor([[0] * target_ids.size(1)] * padding_size).to(target_ids.device)

            decoder_input = torch.cat([true_target_sequence, padding_vector.to(torch.int64)], dim=0)
            # Generate predictions for the current time step
            output_step = self(input_ids, decoder_input)

            # Compute the loss for the current time step
            loss += F.cross_entropy(output_step.transpose(0, 1).reshape(-1, output_step.size(-1)), 
                                decoder_input.transpose(0, 1).reshape(-1), 
                                ignore_index=0)


            predictions = torch.argmax(output_step.squeeze(0), dim=-1)

            # Assuming decoder_input is shape (seq_len, batch_size)
            # Create a mask for non-padding positions
            non_padding_mask = (decoder_input != 0)

            # Flatten the tensors before applying the mask
            predictions_flat = predictions.view(-1)
            decoder_input_flat = decoder_input.view(-1)

            # Only consider non-padding positions for accuracy calculation
            correct_count += (predictions_flat[non_padding_mask.view(-1)] == decoder_input_flat[non_padding_mask.view(-1)]).sum().item()
            total_count += non_padding_mask.sum().item()


        # Average the loss over all time steps
        loss /= target_ids.size(0)

        # Compute accuracy over all time steps
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        '''
        output_reshaped = output.reshape(-1, output.size(-1))
        target_reshaped = target_ids.reshape(-1)

        # Compute loss
        loss = F.cross_entropy(output_reshaped, target_reshaped, ignore_index=0)
        
        # Compute accuracy
        # Take the argmax of the output to get the most likely token
        predictions = torch.argmax(output_reshaped, dim=1)
        # Compute accuracy with masking for padding
        non_padding_mask = (target_reshaped != 0)
        correct_count = ((predictions == target_reshaped) & non_padding_mask).sum().item()
        total_count = non_padding_mask.sum().item()
        # Avoid division by zero
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        accuracy_tensor = torch.tensor(accuracy)
        # Log training loss and accuracy
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.train_step_outputs.append(loss)
        self.train_accuracy_outputs.append(accuracy_tensor)

        return loss



    def validation_step(self, batch, batch_idx):
        input_ids, target_ids, _ = batch

        target_ids = target_ids.permute(1, 0)
        input_ids = input_ids.permute(1, 0)
        
        # Pass the sequence-first tensors to the transformer
        # During validation, the model predicts the output without teacher forcing
        output = self(input_ids, target_ids)

        # Adjust the reshaping to keep the sequence-first format
        output_reshaped = output.reshape(-1, output.size(-1))
        target_reshaped = target_ids.reshape(-1)

        # Compute loss
        loss = F.cross_entropy(output_reshaped, target_reshaped, ignore_index=0)

        # Compute accuracy
        # Take the argmax of the output to get the most likely token
        predictions = torch.argmax(output_reshaped, dim=1)
        # Compute accuracy with masking for padding
        non_padding_mask = (target_reshaped != 0)
        correct_count = ((predictions == target_reshaped) & non_padding_mask).sum().item()
        total_count = non_padding_mask.sum().item()
        # Avoid division by zero
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        accuracy_tensor = torch.tensor(accuracy)

        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)

        #Log accuracy
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.validation_accuracy_outputs.append(accuracy_tensor)

        return loss

    def on_validation_epoch_end(self):

        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()

            epoch_train_acc = torch.stack(self.train_accuracy_outputs).mean()
            self.log("train_accuracy", epoch_train_acc)
            print("train_acc_avg: ", epoch_train_acc)
            self.train_accuracy_outputs.clear()

        if not len(self.validation_step_outputs) == 0:
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            self.log("validation_epoch_average", epoch_average)
            print("val_loss_avg: ", epoch_average)
            self.validation_step_outputs.clear()

            epoch_val_acc = torch.stack(self.validation_accuracy_outputs).mean()
            self.log("validation_accuracy", epoch_val_acc)
            print("val_acc_avg: ", epoch_val_acc)
            self.validation_accuracy_outputs.clear()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-3)
        return optimizer