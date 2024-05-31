import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import math
import os

# Set the TORCH_USE_CUDA_DSA environment variable
os.environ["TORCH_USE_CUDA_DSA"] = "1"

books = pd.read_csv("Books.csv", low_memory=False)
ratings = pd.read_csv("Ratings.csv", low_memory=False)
users = pd.read_csv("Users.csv", low_memory=False)

books_df = pd.read_csv("Books.csv", low_memory=False)
ratings_df = pd.read_csv("Ratings.csv", low_memory=False)
users_df = pd.read_csv("Users.csv", low_memory=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Initialize LabelEncoders for user and item IDs
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit and transform user and item IDs
ratings['user_id'] = user_encoder.fit_transform(ratings['User-ID'])
ratings['item_id'] = item_encoder.fit_transform(ratings['ISBN'])

# Fit and transform user and item IDs
ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['User-ID'])
ratings_df['item_id'] = item_encoder.fit_transform(ratings_df['ISBN'])

ratings_train = ratings[0:919824]
ratings_test = ratings[919825:].reset_index(drop=True)


class RatingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'user_id': self.data['user_id'][idx],
            'book_id': self.data['item_id'][idx],
            'rating': self.data['Book-Rating'][idx]
        }
# Instantiate train and test datasets
train_dataset = RatingDataset(ratings_train)
test_dataset = RatingDataset(ratings_test)

# Create train and test loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(len(train_dataset))
print(len(train_loader))


class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(GMF, self).__init__()
        self.relu = nn.ReLU()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size, 32)
        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        element_product = user_embed * item_embed
        x = self.fc(element_product)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        output = torch.sigmoid(output)  # Ensure output is between 0 and 1
        return output.view(-1)


class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_layers=[64, 32]):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        layers = []
        input_size = embedding_size * 2
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        concat_embed = torch.cat((user_embed, item_embed), dim=1)
        output = self.layers(concat_embed)
        output = torch.sigmoid(output)  # Ensure output is between 0 and 1
        return output.view(-1)


class NCF(nn.Module):
    def __init__(self, gmf_model, mlp_model):
        super(NCF, self).__init__()
        self.gmf = gmf_model
        self.mlp = mlp_model

    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        combined_output = (gmf_output + mlp_output) / 2
        return combined_output


num_users = len(ratings['User-ID'].unique())
num_items = len(ratings['ISBN'].unique())
embedding_size = 64
hidden_layers = [128, 64, 32]
print(num_users)
print(num_items)

# Initialize GMF model
gmf_model = GMF(num_users, num_items, embedding_size).to(device)

# Initialize MLP model
mlp_model = MLP(num_users, num_items, embedding_size, hidden_layers).to(device)

# Loss criterion for GMF and MLP models
models_criterion = nn.MSELoss()

# Optimizer for GMF model
gmf_optimizer = optim.Adam(gmf_model.parameters(), lr=0.001)

# Optimizer for MLP model
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)


def train_gmf(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        i = 0
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['book_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, (ratings.float() / 10))
            loss.backward()
            optimizer.step()

            if (i % 1000 == 0):
                actual_ratings = 10 * predictions
                diff = torch.abs(actual_ratings - ratings).sum().item()
                print(f'Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item()}, Avg. Diff: {(diff / len(ratings))}')

            i = i + 1

            total_loss += loss.item()

        print(f'GMF Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')


def train_mlp(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        i = 0
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['book_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, (ratings.float() / 10))
            loss.backward()
            optimizer.step()

            if (i % 1000 == 0):
                actual_ratings = 10 * predictions
                diff = torch.abs(actual_ratings - ratings).sum().item()
                print(f'Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item()}, Avg. Diff: {(diff / len(ratings))}')

            i = i + 1

            total_loss += loss.item()

        print(f'MLP Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')


num_epochs = 5

# Check if the model parameter files exist
if os.path.isfile('gmf_model.pth') and os.path.isfile('mlp_model.pth'):
    # Load the parameters from files
    gmf_model.load_state_dict(torch.load('gmf_model.pth'))
    mlp_model.load_state_dict(torch.load('mlp_model.pth'))

    # Установка моделей в режим оценки
    gmf_model.eval()
    mlp_model.eval()
else:
    # Train the models if the parameter files don't exist
    print("Training GMF...")
    train_gmf(gmf_model, train_loader, models_criterion, gmf_optimizer, num_epochs)

    print("Training MLP...")
    train_mlp(mlp_model, train_loader, models_criterion, mlp_optimizer, num_epochs)

    # Save the trained model parameters
    torch.save(gmf_model.state_dict(), 'gmf_model.pth')
    torch.save(mlp_model.state_dict(), 'mlp_model.pth')

model = NCF(gmf_model, mlp_model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Проверяем наличие файлов с параметрами моделей
if os.path.isfile('ncf_model.pth'):
    # Если файл существует, загружаем параметры модели из него
    model.load_state_dict(torch.load('ncf_model.pth'))

    # Устанавливаем модель в режим оценки
    model.eval()
else:
    # Если файл не существует, обучаем модель заново
    print("Training NCF...")
    num_epochs = 10
    for epoch in range(num_epochs):
        i = 0
        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['book_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, (ratings.float() / 10))
            loss.backward()
            optimizer.step()

            if (i % 1000 == 0):
                actual_ratings = 10 * predictions
                rmse = math.sqrt(torch.square(actual_ratings - ratings).sum().item() / len(ratings))
                diff = torch.abs(actual_ratings - ratings).sum().item()
                print(
                    f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item()}, Avg. Diff: {(diff / len(ratings))}, RMSE: {rmse}')

            i = i + 1

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Сохраняем обученные параметры модели
    torch.save(model.state_dict(), 'ncf_model.pth')

total_loss = 0
total_diff = 0
total_examples = 0
total_squared_error = 0

with torch.no_grad():
    for batch in test_loader:
        user_ids = batch['user_id'].to(device)
        item_ids = batch['book_id'].to(device)
        ratings = batch['rating'].to(device)

        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, (ratings.float() / 10))

        actual_ratings = 10 * predictions
        diff = torch.abs(actual_ratings - ratings).sum().item()

        total_loss += loss.item() * len(ratings)
        total_diff += diff
        total_examples += len(ratings)
        total_squared_error += torch.square(actual_ratings-ratings).sum().item()

avg_loss = total_loss / total_examples
avg_diff = total_diff / total_examples
rmse = math.sqrt(total_squared_error / total_examples)

print('Evalution Measures:')
print(f'Evaluation Loss: {avg_loss}, Average Difference: {avg_diff}, RMSE: {rmse}')

users.head()
book_ratings = ratings_df.groupby('item_id')['Book-Rating'].mean().reset_index()
book_ratings = book_ratings.sort_values(by='Book-Rating', ascending=False)
top_64_books = book_ratings.head(64)
print(top_64_books)
user_id = 56897
user_id_tensor = torch.LongTensor([user_id] * 64).to(device)

top_64_books = top_64_books['item_id'].tolist()
item_ids_tensor = torch.LongTensor(top_64_books).to(device)

print("User ID Tensor:", user_id_tensor)
print("Top 64 Books Tensor:", item_ids_tensor)

predictions = model(user_id_tensor, item_ids_tensor)

indexed_predictions = [(idx, pred) for idx, pred in enumerate(predictions)]

# Sort the indexed predictions by the prediction values in descending order
sorted_predictions = sorted(indexed_predictions, key=lambda x: x[1], reverse=True)

# Get the top 3 indices
top_3_indices = [idx for idx, _ in sorted_predictions[:3]]

# Get the top 3 book ISBNs
top_3_book_isbns = [item_ids_tensor[idx].item() for idx in top_3_indices]
top_3 = item_encoder.inverse_transform(top_3_book_isbns)

print("Top 3 Book ISBNs:", top_3)
