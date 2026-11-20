import numpy as np

#set random seed
np.random.seed(42)

#define vocabulary
chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

vocab_size=len(chars)
#character to index mapping
char_to_idx={ch: i for i, ch in enumerate(chars)}
# Index-to-character mapping
idx_to_char = {i: ch for i, ch in enumerate(chars)}
#INDEX TO CHARACTER FUNCTION
def one_hot(index):
    vector=np.zeros((vocab_size,1))
    vector[index]=1
    return vector

#create training sequences
seq_length=5
x_train=[]
y_train=[]

for i in range(vocab_size):
    sequence = [
        chars[(i+j) % vocab_size] for j in range(seq_length + 1)

    ]

    inputs=sequence[:-1]
    targets=sequence[1:]

    x_train.append([char_to_idx[ch] for ch in inputs])
    y_train.append([char_to_idx[ch] for ch in targets])

print("vocab size: ", vocab_size)
print("sequence length: ", seq_length)
print("no of training sequence", len(x_train))
print("example input",''.join(chars[i] for i in x_train[0]))
print("example target",''.join(chars[i] for i in y_train[0]))

class vanillaRNN:
    def __init__(
        self, input_size, hidden_size, output_size, learning_rate=0.05
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #initializing weights

        self.U = np.random.randn(hidden_size, input_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(output_size, hidden_size) * 0.01

        # Initialize biases
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((output_size, 1))

    # Softmax function
    def softmax(self, x):

        exp_x = np.exp(x - np.max(x))

        return exp_x / np.sum(exp_x)
#forward propagation
    def forward(self,inputs):
        self.hidden_states={}
        self.outputs={}
        self.probabilities = {}

        # Initial hidden state
        self.hidden_states[-1] = np.zeros((self.hidden_size, 1))

        for t in range(len(inputs)):
            # One-hot encode input
            x = one_hot(inputs[t])

            # Calculate hidden state
            self.hidden_states[t] = np.tanh(
                np.dot(self.U, x) +
                np.dot(self.W, self.hidden_states[t - 1]) +
                self.b
            )

            # Calculate output
            self.outputs[t] = (
                    np.dot(self.V, self.hidden_states[t]) + self.c
            )

            # Apply softmax
            self.probabilities[t] = self.softmax(self.outputs[t])

        return self.probabilities

    def calculate_loss(self,targets):
        loss = 0

        for t in range(len(targets)):
            probability = self.probabilities[t][targets[t], 0]

            loss += -np.log(probability + 1e-12)

        return loss / len(targets)

    def backward(self, inputs, targets):

        # Initialize gradients
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)

        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        dh_next = np.zeros((self.hidden_size, 1))

        # Iterate backwards through time
        for t in reversed(range(len(inputs))):

            # Output gradient
            dy = self.probabilities[t].copy()

            dy[targets[t]] -= 1

            # Output layer gradients
            dV += np.dot(dy, self.hidden_states[t].T)
            dc += dy

            # Hidden state gradient
            dh = np.dot(self.V.T, dy) + dh_next

            # Tanh derivative
            da = dh * (1 - self.hidden_states[t] ** 2)

            # Hidden layer gradients
            x = one_hot(inputs[t])

            dU += np.dot(da, x.T)

            dW += np.dot(
                da,
                self.hidden_states[t-1].T
            )

            db += da

            # Pass gradient to previous time step
            dh_next = np.dot(self.W.T, da)

        # Gradient clipping
        for gradient in [dU, dW, dV, db, dc]:

            np.clip(gradient, -5, 5, out=gradient)

        return dU, dW, dV, db, dc
    def update_weights(self, gradients):

        dU, dW, dV, db, dc = gradients

        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.V -= self.learning_rate * dV

        self.b -= self.learning_rate * db
        self.c -= self.learning_rate * dc

    def train(self, X_train, Y_train, epochs=1000):

        for epoch in range(epochs):

            total_loss = 0

            for inputs, targets in zip(X_train, Y_train):

                # Forward propagation
                self.forward(inputs)

                # Calculate loss
                loss = self.calculate_loss(targets)

                total_loss += loss

                # Backpropagation
                gradients = self.backward(inputs, targets)

                # Update weights
                self.update_weights(gradients)

            average_loss = total_loss / len(X_train)

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch}, Loss: {average_loss:.4f}"
                )


    def predict(self, input_sequence):

        inputs = [char_to_idx[ch] for ch in input_sequence]

        probabilities = self.forward(inputs)

        predictions = []

        for t in range(len(inputs)):

            predicted_index = np.argmax(probabilities[t])

            predictions.append(idx_to_char[predicted_index])

        return ''.join(predictions)
# Create RNN model
model = vanillaRNN(
    input_size=26,
    hidden_size=64,
    output_size=26,
    learning_rate=0.05
)

# Train model
model.train(
    x_train,
    y_train,
    epochs=1000
)

# Test model
test_sequence = "ABCDE"

prediction = model.predict(test_sequence)

print("\nInput sequence:", test_sequence)
print("Predicted next characters:", prediction)
print("Expected output: BCDEF")


