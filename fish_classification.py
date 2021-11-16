import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
image_dir = Path("./Fish_Dataset")

# 获取 鱼的图片路径和对应标签
filePaths = list(image_dir.glob(r'**/*.png'))

print(filePaths[:10])
print(len(filePaths))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filePaths))

filePaths = pd.Series(filePaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# 拼接鱼类图片 及其 对应标签
image_df = pd.concat([filePaths, labels], axis=1)

# 查看数据集的9幅图及其标签
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 7), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[i * 1000 + 1]))
    ax.set_title(image_df.Label[i * 1000 + 1])

plt.tight_layout()
plt.show()
# 乱序 数据集 ，重设索引
image_df = image_df.sample(frac=1).reset_index(drop=True)

# 分成 训练集，测试集
train_df, test_df = train_test_split(image_df, train_size=0.8, shuffle=True, random_state=42)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Label',
    target_size=(224, 224), color_mode='rgb', class_mode='categorical',
    batch_size=4, shuffle=True, seed=42, subset='training')
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Label',
    target_size=(224, 224), color_mode='rgb', class_mode='categorical',
    batch_size=4, shuffle=True, seed=42, subset='validation')
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df, x_col='Filepath', y_col='Label',
    target_size=(224, 224), color_mode='rgb', class_mode='categorical',
    batch_size=4, shuffle=True)


def create_model(input_shape=(224, 224, 3)):
    inputs = Input(input_shape)
    base_model = EfficientNetB1(input_shape=input_shape, include_top=False, classes=9)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(56, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(9, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


model = create_model((224, 224, 3))
metrics = ['accuracy', 'AUC']
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=metrics)

checkpoint_save_path = './models/EfficientNetB1_model.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(monitor='val_loss', mode='min', filepath=checkpoint_save_path, verbose=1,
                    save_best_only=True, save_weights_only=False)
]
if os.path.exists(checkpoint_save_path):
    print("--------------load the model------------------")
    model.load_weights(checkpoint_save_path)
history = model.fit(train_images, batch_size=32, validation_data=val_images, epochs=10, callbacks=callbacks)
model.summary()
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="accuracy")
plt.plot(history.history['val_accuracy'], label="val_accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()

results = model.evaluate(test_images, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
