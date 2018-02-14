# Variational AutoEncoder

This was originally based on the [keras script for VAE's](https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py).
I modified it to be object oriented and trainable in a very simple way.

I used it originally for my Reinforcement Learning class when implementing a Model-Free Episodic learner. 

## Usage
### The easy way
Run train.py with the correct arguments!

### In-Depth Usage
Usage example:
```python
from model import VariationalAutoencoder
vae = VariationalAutoencoder((im_w, im_h), 3)

# Modify the VAE variables here
vae.latent_dim = 2

# Train the VAE
vae.train(arr_of_train_images, arr_of_test_images)
```

The trained encoder and decoder will be automatically saved. In order to load the model and use it, simple:
```python
from keras.models import load_model

encoder = load_model("encoder_file.h5")
decoder = load_model("decoder_file.h5")

encoded = encoder.predict(img)
decoded = decoder.predict(encoded)
```