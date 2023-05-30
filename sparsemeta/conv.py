import haiku as hk
import math

from jax import nn


class ConvBlock(hk.Module):
    def __init__(self, channels, name=None, norm="batch_norm"):
        super().__init__(name=name)
        self.channels = channels
        self.norm = norm

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = hk.Conv2D(self.channels, kernel_shape=3,
            stride=1, with_bias=True, name='conv')(outputs)
        if self.norm == 'batch_norm':
            outputs = hk.BatchNorm(create_scale=True, create_offset=True,
                decay_rate=0.9, name='norm')(outputs, is_training)
        elif self.norm == "layer_norm":
            outputs = hk.LayerNorm(  # Normalize the features
                axis=1,
                create_scale=True,
                create_offset=True
            )(outputs)
        elif self.norm is not None:
            raise NotImplementedError(f"--conv_norm {self.norm} is not implemented.")
        outputs = nn.relu(outputs)
        outputs = hk.max_pool(outputs, 2, 2, padding='VALID')
        return outputs


class Conv4(hk.Module):
    def __init__(self, num_filters=64, normalize_outputs=False, name=None, norm="batch_norm"):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.normalize_outputs = normalize_outputs
        self.norm = norm

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = ConvBlock(self.num_filters, name='layer1', norm=self.norm)(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer2', norm=self.norm)(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer3', norm=self.norm)(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer4', norm=self.norm)(outputs, is_training)
        outputs = outputs.reshape(inputs.shape[:-3] + (-1,))
        normalization = math.sqrt(outputs.shape[-1]) if self.normalize_outputs else 1.
        return outputs / normalization


class ConvDisentanglement(hk.Module):
    def __init__(self, z_dim, name=None):
        super().__init__(name=name)
        self.z_dim = z_dim

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = hk.Conv2D(32, kernel_shape=4, stride=2, with_bias=True, name='conv1')(outputs)
        outputs = nn.relu(outputs)
        outputs = hk.Conv2D(32, kernel_shape=4, stride=2, with_bias=True, name='conv2')(outputs)
        outputs = nn.relu(outputs)
        outputs = hk.Conv2D(64, kernel_shape=2, stride=2, with_bias=True, name='conv3')(outputs)
        outputs = nn.relu(outputs)
        outputs = hk.Conv2D(64, kernel_shape=2, stride=2, with_bias=True, name='conv4')(outputs)
        outputs = nn.relu(outputs)
        outputs = outputs.reshape(inputs.shape[:-3] + (-1,))
        outputs = hk.Linear(256)(outputs)
        outputs = nn.relu(outputs)
        outputs = hk.Linear(self.z_dim)(outputs)
        return outputs
