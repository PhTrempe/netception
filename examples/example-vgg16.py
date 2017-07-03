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
