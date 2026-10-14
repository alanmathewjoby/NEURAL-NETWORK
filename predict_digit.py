
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter


model = np.load("digit_model.npz")

W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]


def preprocess_image(path):
    original = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(original)

    gray = ImageOps.autocontrast(gray)

    threshold = 180
    mask = gray.point(
        lambda pixel: 255 if pixel < threshold else 0
    )

    bbox = mask.getbbox()

    if bbox is None:
        raise ValueError(
            "No dark digit found. Use a clear image "
            "of black ink on white paper."
        )

    digit = gray.crop(bbox)

    digit = ImageOps.invert(digit)
    
    width, height = digit.size
    side = max(width, height) + max(2, int(max(width, height) * 0.2))

    canvas = Image.new("L", (side, side), 0)
    x = (side - width) // 2
    y = (side - height) // 2
    canvas.paste(digit, (x, y))

    small = canvas.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )


    pixels = np.asarray(small, dtype=np.float64) / 255.0

    return pixels.reshape(1, 64), small


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return softmax(z2)



image_path = "digit.png"

X_image, processed_image = preprocess_image(image_path)
probabilities = predict(X_image)[0]

predicted_digit = int(np.argmax(probabilities))
confidence = float(probabilities[predicted_digit]) * 100

print("Predicted digit:", predicted_digit)
print(f"Model confidence: {confidence:.2f}%")

print("\nProbabilities:")
for digit, probability in enumerate(probabilities):
    print(f"{digit}: {probability * 100:6.2f}%")



plt.figure(figsize=(7, 3))

plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path))
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_image, cmap="gray", vmin=0, vmax=255)
plt.title("Processed 8 x 8 input")
plt.axis("off")

plt.tight_layout()
plt.show()
