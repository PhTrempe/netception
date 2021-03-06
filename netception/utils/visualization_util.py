import numpy as np


class VisualizationUtil(object):
    """
    A utility class which provides visualization-related functionality.
    """

    @staticmethod
    def inception_to_bytes(inception, colorfulness=0.2):
        """
        Converts inception data to image bytes.

        :param inception: The inception data to convert.
        :param colorfulness: The colorfulness of the generated bytes.

        :return: image_bytes

        :raises: ValueError
        """
        if not 0.0 <= colorfulness <= 1.0:
            raise ValueError("colorfulness must be in [0.0; 1.0]")
        if not isinstance(inception, np.ndarray):
            raise ValueError("inception is not a numpy.ndarray")

        a = np.array(inception)
        a_0_centered = a - a.mean()
        k = 2 * max(abs(a.max()), abs(a.min()))
        a_normalized = a_0_centered / k
        epsilon = np.finfo(float).eps
        a_std_01 = a_normalized / (a_normalized.std() + epsilon) * colorfulness
        a_clip_0_1 = np.clip(a_std_01 + 0.5, 0.0, 1.0)
        return (a_clip_0_1 * 255).astype("uint8")
