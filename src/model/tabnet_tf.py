"""
TabNet implementation in TensorFlow/Keras for Telco Customer Churn project.
Reference: https://arxiv.org/abs/1908.07442
If you want a more advanced or official implementation, consider using tf-keras-tabnet or adapt this as needed.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GLULayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        linear, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        return linear * tf.sigmoid(gate)

class FeatureTransformer(layers.Layer):
    def __init__(self, units, n_glu, **kwargs):
        super().__init__(**kwargs)
        self.glu_layers = [GLULayer(units) for _ in range(n_glu)]
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.bn(inputs)
        for glu in self.glu_layers:
            x = glu(x)
        return x

class AttentiveTransformer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, prior):
        x = self.bn(inputs)
        x = self.dense(x)
        x = tf.nn.softmax(x * prior, axis=1)
        return x

class TabNet(tf.keras.Model):
    def __init__(self, feature_dim, output_dim, n_steps=3, n_glu=2, decision_dim=8, relaxation_factor=1.5, **kwargs):
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.decision_dim = decision_dim
        self.relaxation_factor = relaxation_factor
        self.n_glu = n_glu

        self.initial_transformer = FeatureTransformer(decision_dim, n_glu)
        self.attentive_transformers = [AttentiveTransformer(feature_dim) for _ in range(n_steps)]
        self.feature_transformers = [FeatureTransformer(decision_dim, n_glu) for _ in range(n_steps)]
        self.output_layer = layers.Dense(output_dim, activation="softmax")

    def call(self, inputs):
        prior = tf.ones_like(inputs)
        M_loss = 0.0
        x = self.initial_transformer(inputs)
        aggregated = 0.0
        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](x, prior)
            x = mask * inputs
            x = self.feature_transformers[step](x)
            aggregated += x
            prior *= self.relaxation_factor - mask
        out = self.output_layer(aggregated)
        return out

# Example usage (to be replaced by pipeline):
# model = TabNet(feature_dim=..., output_dim=2)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, ...)
