
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

X = digits.data.astype(np.float64) / 16.0
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Total images:", len(X))
print("Training images:", len(X_train))
print("Testing images:", len(X_test))



fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()




def one_hot(y, classes=10):
    result = np.zeros((len(y), classes))
    result[np.arange(len(y)), y] = 1
    return result


Y_train = one_hot(y_train)




np.random.seed(42)

input_size = 64
hidden_size = 128
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(
    2 / input_size
)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(
    1 / hidden_size
)
b2 = np.zeros((1, output_size))



def relu(z):
    return np.maximum(0, z)


def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)




def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    probabilities = softmax(z2)

    return z1, a1, z2, probabilities



def cross_entropy(probs, Y):
    eps = 1e-12
    return -np.mean(
        np.sum(Y * np.log(probs + eps), axis=1)
    )



def train_step(X_batch, Y_batch, learning_rate):
    global W1, b1, W2, b2


    z1, a1, z2, probs = forward(X_batch)


    loss = cross_entropy(probs, Y_batch)
    m = X_batch.shape[0]

    dz2 = (probs - Y_batch) / m
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X_batch.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)


    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return loss




def accuracy(X, y):
    _, _, _, probs = forward(X)
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == y)




epochs = 1000
batch_size = 64
learning_rate = 0.1

loss_history = []

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    batch_losses = []

    for start in range(0, len(X_train), batch_size):
        xb = X_shuffled[start:start + batch_size]
        yb = Y_shuffled[start:start + batch_size]

        loss = train_step(xb, yb, learning_rate)
        batch_losses.append(loss)

    epoch_loss = np.mean(batch_losses)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train accuracy: {accuracy(X_train, y_train):.3f} | "
            f"Test accuracy: {accuracy(X_test, y_test):.3f}"
        )



plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Neural Network Learning from Scratch")
plt.grid(True)
plt.show()



print("\nFinal test accuracy:", accuracy(X_test, y_test))



def show_predictions(X, y, count=10):
    _, _, _, probs = forward(X[:count])
    predictions = np.argmax(probs, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flat):
        image = X[i].reshape(8, 8)

        ax.imshow(image, cmap="gray")
        confidence = probs[i, predictions[i]] * 100

        ax.set_title(
            f"Pred: {predictions[i]} | Actual: {y[i]}\n"
            f"Confidence: {confidence:.1f}%"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


show_predictions(X_test, y_test)



np.savez(
    "digit_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("Model saved to digit_model.npz")

