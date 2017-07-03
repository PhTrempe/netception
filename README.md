<div align="center">
    <img src="http://gdurl.com/g-tt" width="540">
</div>

# Netception

## Description

Netception is a neural network inception library developed 
in Python 3 and used with Keras.

## How to Install

```
pip install netception
```

## Dependencies

Here are Netception's required dependencies.

* `numpy`
* `keras`

Here are some optional dependencies one might need.

* `h5py` (to load and save Keras models stored in files)
* `pillow` (to manipulate images)

Also note that Keras requires a backend library like TensorFlow to operate.

```
pip install tensorflow
```

If GPU support is desired, it is also possible to use the GPU version of 
TensorFlow.

```
pip install tensorflow-gpu
```

## An Exhaustively Commented Example to Get You Started

```python
import os

from PIL import Image
from keras import applications, backend

from netception.inceptor import Inceptor
from netception.utils.visualization_util import VisualizationUtil


if __name__ == "__main__":
    # Load the model to incept
    # (Here, we load the pretrained VGG16 model from Keras)
    model = applications.VGG16()

    # Print the model's summary to see its layers
    model.summary()

    # Determine the target to incept within the model
    # (Here, we choose to incept the output of the 455th filter of the
    # convolutional layer "block5_conv3")
    target = model.get_layer("block5_conv3").output[:, :, :, 455]

    # Create an inceptor and configure it
    # (Here, we create an inceptor with our model and target. We also set an
    # inception rate of 0.25, a maximal number of steps of 50, and parameters
    # for early stopping if the inception score stops improving enough)
    inceptor = Inceptor(
        model=model,
        target=target,
        inception_rate=0.5,
        max_steps=200,
        improvement_check_interval=5,
        improvement_threshold=0.05
    )

    # Run the inceptor
    inception, score = inceptor.incept()

    # Convert the resulting inception into image data
    image_data = VisualizationUtil.inception_to_bytes(
        inception=inception,
        colorfulness=0.15
    )

    # Create an image from the image data, and resize the image
    image = Image.fromarray(image_data).resize((512, 512), Image.BICUBIC)

    # Show the image
    image.show()

    # Save the image
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image.save(os.path.join(script_dir, "inception.png"))

    # Clear the backend session
    backend.clear_session()

```

This is what the result looks like.

<div align="center">
    <img src="http://gdurl.com/XWya" width="512">
</div>

## The Same Example In Compact Form For a Quick Copy & Paste

```python
import os

from PIL import Image
from keras import applications, backend

from netception.inceptor import Inceptor
from netception.utils.visualization_util import VisualizationUtil


if __name__ == "__main__":
    model = applications.VGG16()
    model.summary()
    target = model.get_layer("block5_conv3").output[:, :, :, 455]
    inceptor = Inceptor(model, target, 0.5, 200, 5, 0.05)
    inception, score = inceptor.incept()
    image_data = VisualizationUtil.inception_to_bytes(inception, 0.15)
    image = Image.fromarray(image_data).resize((512, 512), Image.BICUBIC)
    image.show()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image.save(os.path.join(script_dir, "inception.png"))
    backend.clear_session()

```
