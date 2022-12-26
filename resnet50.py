from tensorflow.keras.layers import Dropout,Dense, Flatten, Input,AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import tensorflow as tf

# 91mb
def create_model_resnet50(classes):
  basemodel = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    input_tensor=Input((224,224,3)),
    weights='imagenet',
  )
  fc = basemodel.output
  fc = AveragePooling2D(pool_size=(7, 7))(fc)
  fc = Flatten(name="flatten")(fc)
  fc = Dense(256, activation="relu")(fc)
  fc = Dropout(0.5)(fc)
  fc = Dense(classes, activation="sigmoid")(fc)
  return Model(inputs=basemodel.input,outputs=fc)


# 16mb
def create_model_mobilenet(classes):
    basemodel = tf.keras.applications.MobileNet(
        depth_multiplier=1,
        include_top=False,
        weights="imagenet",
        input_tensor=Input((224,224,3)),
    )
    fc = basemodel.output
    fc = AveragePooling2D(pool_size=(7, 7))(fc)
    fc = Flatten(name="flatten")(fc)
    fc = Dense(256, activation="relu")(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(classes, activation="sigmoid")(fc)
    return Model(inputs=basemodel.input,outputs=fc)

# 800mb
def create_model_mobilenet(classes):
    basemodel = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=Input((224,224,3)),
    )
    fc = basemodel.output
    fc = AveragePooling2D(pool_size=(7, 7))(fc)
    fc = Flatten(name="flatten")(fc)
    fc = Dense(256, activation="relu")(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(classes, activation="sigmoid")(fc)
    return Model(inputs=basemodel.input,outputs=fc)
