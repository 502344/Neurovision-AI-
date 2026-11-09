"""Keras model builders shared by training, feature extraction, Grad-CAM and the API.

Design notes
* Models accept raw RGB pixels in [0, 255]. Backbone-specific scaling is *inside* the model.
* The CNN is built as a flat functional graph so Grad-CAM can reach its last conv layer.
* The penultimate feature layers are named ``cnn_features`` / ``vit_features``.
"""
from __future__ import annotations

import keras
from keras import layers

from ml import config

LAST_CONV_LAYER = {"resnet50": "conv5_block3_out", "efficientnet": "top_activation"}


@keras.saving.register_keras_serializable(package="neurovision")
class CaffePreprocess(layers.Layer):
    """ResNet50 (Keras/caffe) preprocessing: RGB->BGR and ImageNet mean subtraction."""

    def call(self, x):
        x = keras.ops.flip(x, axis=-1)
        mean = keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)
        return x - mean

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package="neurovision")
class ClsToken(layers.Layer):
    """Select the [CLS] token (index 0) from a ViT token sequence."""

    def call(self, x):
        return x[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def augmentation_block(name="augmentation") -> keras.Sequential:
    """Mild augmentation, active only during training (Keras disables it at inference)."""
    aug = []
    if config.AUGMENT_HORIZONTAL_FLIP:
        aug.append(layers.RandomFlip("horizontal"))
    aug += [
        layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.0),   # about +/- 11 degrees
        layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
        layers.RandomBrightness(0.08, value_range=(0, 255)),
        layers.RandomContrast(0.08),
    ]
    return keras.Sequential(aug, name=name)


def _head(x, prefix: str):
    x = layers.Dropout(config.DROPOUT, name=f"{prefix}_drop1")(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu", name=f"{prefix}_dense")(x)
    x = layers.Dropout(config.DROPOUT, name=f"{prefix}_drop2")(x)
    return layers.Dense(1, activation="sigmoid", name="asd_probability")(x)


def build_cnn_classifier(backbone: str | None = None, pretrained: bool | None = None,
                         image_size: int | None = None) -> keras.Model:
    """Backbone -> GlobalAveragePooling ("cnn_features") -> Dense -> sigmoid (P(ASD))."""
    backbone = (backbone or config.CNN_BACKBONE).lower()
    pretrained = config.USE_PRETRAINED if pretrained is None else pretrained
    size = image_size or config.IMAGE_SIZE
    weights = "imagenet" if pretrained else None
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation_block()(inp)
    if backbone == "resnet50":
        x = CaffePreprocess(name="resnet_preprocess")(x)
        base = keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=x)
    elif backbone in ("efficientnet", "efficientnetb0"):
        backbone = "efficientnet"
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=x)
    else:
        raise ValueError(f"Unknown CNN backbone '{backbone}' (use resnet50 or efficientnet)")
    feat = layers.GlobalAveragePooling2D(name=config.CNN_FEATURE_DIM_LAYER)(base.output)
    out = _head(feat, "cnn")
    model = keras.Model(inp, out, name=f"{backbone}_asd_classifier")
    model.backbone_layer_names = [l.name for l in base.layers]
    return model


def build_vit_classifier(pretrained: bool | None = None, image_size: int | None = None) -> keras.Model:
    """ViT backbone -> [CLS] token ("vit_features") -> Dense -> sigmoid (P(ASD))."""
    import keras_hub

    pretrained = config.USE_PRETRAINED if pretrained is None else pretrained
    size = image_size or config.IMAGE_SIZE
    if pretrained and config.VIT_PRESET:
        if size != 224:
            raise ValueError("The pretrained ViT preset requires IMAGE_SIZE=224.")
        backbone = keras_hub.models.ViTBackbone.from_preset(config.VIT_PRESET)
    else:
        backbone = keras_hub.models.ViTBackbone(image_shape=(size, size, 3), **config.VIT_TINY)
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation_block()(inp)
    # NOTE: keras-hub ViT ImageNet presets use mean=std=0.5 scaling ([-1, 1]).
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="vit_rescale")(x)
    tokens = backbone(x)
    feat = ClsToken(name=config.VIT_FEATURE_LAYER)(tokens)
    out = _head(feat, "vit")
    model = keras.Model(inp, out, name="vit_asd_classifier")
    model.vit_backbone = backbone
    return model


def load_keras_model(path) -> keras.Model:
    """Load a saved .keras model (registers keras-hub + custom layers first)."""
    try:
        import keras_hub  # noqa: F401  (registers ViTBackbone for deserialisation)
    except Exception:
        pass
    return keras.models.load_model(str(path), compile=False)


def feature_extractor(model: keras.Model, layer_name: str) -> keras.Model:
    return keras.Model(model.inputs, model.get_layer(layer_name).output, name=f"{layer_name}_extractor")
