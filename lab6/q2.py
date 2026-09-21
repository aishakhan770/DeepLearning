#tensors
import torch
x=torch.tensor([1,2,3]) #create a tensor
print(x)
print(x.shape)
print(x.dtype)

#2D tensor
x=torch.tensor([
    [1,2],
    [3,4]
])
print(x)
print(x.shape)

#random tensors
x=torch.randn(2,3)
print(x)

#tensor operations
x=torch.tensor([1.,2.,3.])
print(x+2)

#matrix multiplications
a=torch.tensor([
    [1,2],
    [3,4]
])

b=torch.tensor([
    [5,6],
    [7,8]
])
c=a @ b
print(c)

#tensor indexing
x = torch.tensor([
    [10, 20],
    [30, 40]
])

print(x[0])
print(x[0, 1])

#dataset
import torch
from torch.utils.data import TensorDataset

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

dataset = TensorDataset(X, y)

print(len(dataset))
print(dataset[0])

#dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
for X_batch, y_batch in dataloader:

    print("X:")
    print(X_batch)

    print("y:")
    print(y_batch)

#transforms
from torchvision import transforms
transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
    ])

#normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

#build model
from torch import nn
model = nn.Sequential(

    # 2 inputs → 4 neurons
    nn.Linear(2, 4),

    # Activation function
    nn.Sigmoid(),

    # 4 neurons → 1 output neuron
    nn.Linear(4, 1),

    # Output activation
    nn.Sigmoid()
)


print("\nModel:")
print(model)

#loss function
loss_function = nn.BCELoss() # BCELoss = Binary Cross Entropy Loss

# It measures how different the prediction is
# from the actual answer.

#optimizer

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.5
)

#training
for epoch in range(5000):
    for X_batch, y_batch in dataloader:
        prediction = model(X_batch)
        loss = loss_function(
            prediction,
            y_batch
        )
        optimizer.zero_grad()  # PyTorch accumulates gradients.
        # Therefore we clear the old gradients first.

        loss.backward() #autograd
        #update weights
        optimizer.step()

        if epoch % 500 == 0:

         print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item()
        )
 #TEST THE TRAINED MODEL
with torch.no_grad():

    prediction = model(X)
print("FINAL PREDICTIONS")
print("\nActual:")
print(y)

print("\nPrediction:")
print(prediction)

print("\nRounded prediction:")
print(torch.round(prediction))
#save the model
torch.save(
    model.state_dict(),
    "xor_model.pth"
)

print("\nModel saved!")

#load the model. save the model architecture again
loaded_model = nn.Sequential(

    nn.Linear(2, 4),
    nn.Sigmoid(),

    nn.Linear(4, 1),
    nn.Sigmoid()
)
# Load the saved weights and biases

loaded_model.load_state_dict(
    torch.load(
        "xor_model.pth",
        weights_only=True
    )
)


# Put the model into evaluation mode

loaded_model.eval()

#test the loaded model
with torch.no_grad():

    loaded_prediction = loaded_model(X)


print("\nPrediction from loaded model:")

print(loaded_prediction)

print("\nRounded prediction:")

print(torch.round(loaded_prediction))

