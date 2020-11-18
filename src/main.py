import torch
from torch.nn import Sequential, Linear, ReLU, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
from tqdm import trange

from src.data.ucmerced_dataset import UcMercedDataset
from src.settings import DATA_DIRECTORY

dset = UcMercedDataset(DATA_DIRECTORY)

dataloader = DataLoader(dset, batch_size=50,
                        shuffle=True, num_workers=0)


model = Sequential(
    Conv2d(3, 32, (3, 3)),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(32, 32, (3, 3)),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(32, 64, (3, 3)),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(57600, 100),
    ReLU(),
    Linear(100, 50)
)

model = model.cuda()
# optim = torch.optim.Adam(model.parameters())
# criterion = torch.nn.TripletMarginLoss()
#
#
# t = trange(3)
# for epoch in t:
#     loss_sum = 0.0
#     for i_batch, sample_batched in enumerate(dataloader):
#         optim.zero_grad()
#         anchors = sample_batched['a'].permute(0,3,1,2).float().cuda()
#         positives = sample_batched['p'].permute(0,3,1,2).float().cuda()
#         negatives = sample_batched['n'].permute(0,3,1,2).float().cuda()
#         a = model(anchors)
#         p = model(positives)
#         n = model(negatives)
#         loss = criterion(a, p, n)
#         loss_sum += loss.item()
#         loss.backward()
#         optim.step()
#         t.set_description(f"Batch: {i_batch}, loss: {loss.item()}")
#         t.refresh()
#     print(f"Epoch: {epoch}, Loss: {loss_sum}")

paths = []
embeddings = []
with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader):
        anchors = sample_batched['a'].permute(0, 3, 1, 2).float().cuda()
        anchor_paths = sample_batched['path']
        paths.extend(anchor_paths)
        a = model(anchors).cpu()
        embeddings.append(a)

    embeddings = torch.cat(embeddings)

example = dset[0]
x, y, path = example['a'], example['y_a'], example['path']
example_embedding = model(x)
distances = torch.dist(example_embedding, embeddings)
sorting = torch.argsort(distances)