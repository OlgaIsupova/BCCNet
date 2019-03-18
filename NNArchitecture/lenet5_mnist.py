import tensorflow as tf


def tf_cross_entropy_with_logits(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def cnn_for_mnist(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(10))

    adam = tf.keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss=tf_cross_entropy_with_logits, metrics=['accuracy'])

    return cnn_model

