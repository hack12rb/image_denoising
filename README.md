# Image Denoising using Convolutional Neural Networks

## Project Description

This project focuses on image denoising using Convolutional Neural Networks (CNNs). The goal is to reduce noise from images and recover the clean image. The implemented model is trained and evaluated using a dataset of noisy and clean image pairs.

## Architecture

The model uses a CNN architecture with multiple convolutional and upsampling layers. The architecture effectively captures and removes noise from the input images. Key specifications include:
- **Framework:** TensorFlow/Keras
- **Layers:** Multiple convolutional layers with LeakyReLU activation and upsampling layers.
- **Optimizer:** Adam
- **Loss Function:** Binary_crossentropy

The model achieved an average PSNR (Peak Signal-to-Noise Ratio) of 17.62 dB.

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or later
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/image-denoising-cnn.git
    cd image-denoising-cnn
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow numpy matplotlib
    ```



## Usage

### Training the Model

1. Load and preprocess the dataset:
    ```python
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    train_ds = image_dataset_from_directory(
        'path/to/train',
        image_size=(256, 256),
        batch_size=32,
        label_mode=None
    )

    test_ds = image_dataset_from_directory(
        'path/to/test',
        image_size=(256, 256),
        batch_size=32,
        label_mode=None
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x: normalization_layer(x))
    test_ds = test_ds.map(lambda x: normalization_layer(x))
    ```

2. Define and compile the model:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, UpSampling2D

    model = Sequential([
        Conv2D(32, (3, 3), activation=LeakyReLU(), padding='same', input_shape=(256, 256, 3)),
        Conv2D(32, (3, 3), activation=LeakyReLU(), padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation=LeakyReLU(), padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    ```

3. Train the model:
    

### Evaluating the Model

1. Calculate PSNR for evaluation:
    ```python
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import numpy as np

    noisy_images = next(iter(train_noisy_ds)).numpy()
    clean_images = next(iter(train_ds)).numpy()
    denoised_images = model.predict(noisy_images)

    psnr_values = [psnr(clean_images[i], denoised_images[i]) for i in range(len(clean_images))]
    average_psnr = np.mean(psnr_values)
    print(f'Average PSNR: {average_psnr}')
    ```


            
           
           
    

## Summary and Future Work

### Findings
- The CNN-based model effectively reduces noise in images.
- Achieved an average PSNR of 17.62 dB.

### Improvements
- Experiment with deeper networks or adding skip connections.
- Use data augmentation to improve robustness.
- Implement advanced loss functions like Structural Similarity Index (SSIM) for better perceptual quality.



