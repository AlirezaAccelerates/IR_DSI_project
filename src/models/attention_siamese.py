import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from neural_inverted_index_dsi.src.utils.losses import ContrastiveLoss

class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.W_q = nn.Linear(input_size, input_size)
        self.W_k = nn.Linear(input_size, input_size)
        self.W_v = nn.Linear(input_size, input_size)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, query, key, value, mask=None):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights.unsqueeze(-2), v).squeeze(-2)
        output = self.fc(attended_values)

        return output.squeeze(-1), attention_weights


class CustomTransformer(nn.Module):
    def __init__(self, embedding_size):
        super(CustomTransformer, self).__init__()
        self.attention = AttentionLayer(embedding_size)
        self.feedforward = nn.Sequential(
            nn.Linear(840, 480),
            nn.ReLU(),
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_embeddings, query_mask=None):

        x = input_embeddings

        # Self-attention layer
        attended, attention_weights = self.attention(x, x, x, query_mask)

        # Residual connection and dropout
        x = x + self.dropout(attended)

        output = self.feedforward(x)

        return output


class SiameseTransformer(pl.LightningModule):
    def __init__(self, embedding_size):
        super(SiameseTransformer, self).__init__()
        self.embedding_size = embedding_size 

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_accuracy_outputs = []
        self.transformer = CustomTransformer(embedding_size)
        self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, query_embeddings, document_embeddings, query_mask=None):
        query_output = self.transformer(query_embeddings, query_mask)
        document_output = self.transformer(document_embeddings)

        similarity = F.cosine_similarity(query_output, document_output, dim=1)

        return similarity


    def training_step(self, batch, batch_idx):
        # Get the query, document and relevance label from the batch
        query, document, relevance = batch

        similarity = self(query, document).squeeze()

        loss = self.criterion(similarity, relevance)
        self.train_step_outputs.append(loss)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        accuracy = self.calculate_accuracy(similarity, relevance)

        return loss

    def validation_step(self, batch, batch_idx):
        # Get the query, document and relevance label from the batch
        query, document, relevance = batch

        # Forward pass
        similarity = self(query, document).squeeze()
        # Calculate binary cross-entropy loss
        loss = self.criterion(similarity, relevance)
        self.validation_step_outputs.append(loss)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        accuracy = self.calculate_accuracy(similarity, relevance)
        self.validation_accuracy_outputs.append(accuracy)

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
            accuracy_avg = sum(self.validation_accuracy_outputs) / len(self.validation_accuracy_outputs)
            print("accuracy: ", accuracy_avg)
            self.validation_accuracy_outputs.clear()

    def calculate_accuracy(self, predictions, labels):
        predictions = (predictions > 0.5).float()  # Assuming binary classification
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SiameseTransformerContrastive(pl.LightningModule):
    def __init__(self, embedding_size):
        super(SiameseTransformerContrastive, self).__init__()
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_accuracy_outputs = []

        self.transformer = CustomTransformer(embedding_size)

        self.criterion = ContrastiveLoss()

    def forward(self, query_embeddings, document_embeddings, query_mask=None):
        query_output = self.transformer(query_embeddings, query_mask)
        document_output = self.transformer(document_embeddings)

        return query_output, document_output


    def training_step(self, batch, batch_idx):
        # Get the query, document and relevance label from the batch
        query, document, relevance = batch

        # Forward pass
        query_output, document_output = self(query, document)
        # Compute loss
        loss = self.criterion(query_output.squeeze(), document_output.squeeze(), relevance)
        # Append loss to list
        self.train_step_outputs.append(loss)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        query, document, relevance = batch

        # Forward pass
        query_output, document_output = self(query, document)
        # Compute loss
        loss = self.criterion(query_output.squeeze(), document_output.squeeze(), relevance)
        # Append loss to list
        self.validation_step_outputs.append(loss)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        # Compute accuracy
        similarity = F.cosine_similarity(query_output, document_output, dim=1)
        accuracy = self.calculate_accuracy(similarity, relevance)
        self.validation_accuracy_outputs.append(accuracy)

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
            accuracy_avg = sum(self.validation_accuracy_outputs) / len(self.validation_accuracy_outputs)
            print("accuracy: ", accuracy_avg)
            self.validation_accuracy_outputs.clear()

    def calculate_accuracy(self, predictions, labels):
        predictions = (predictions > 0.5).float()  # Assuming binary classification
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)