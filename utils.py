import tensorflow as tf


def generator(inputs_z, training=True, reuse=False):
    """
    :param inputs_z:随机生成的噪声
    :param training: BN层是否训练
    :param reuse: 是否重用变量中的参数
    :return: 生成的图像，值为（-1,1）
    """
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.dense(inputs_z, 1024 * 4 * 4)
        x = tf.reshape(x, [-1, 4, 4, 1024])
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training))

        x = tf.layers.conv2d_transpose(x, 512, 5, 2, 'same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training))

        x = tf.layers.conv2d_transpose(x, 256, 5, 2, 'same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training))

        x = tf.layers.conv2d_transpose(x, 128, 5, 2, 'same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=training))

        x = tf.layers.conv2d_transpose(x, 3, 5, 2, 'same')
        x = tf.nn.tanh(tf.layers.batch_normalization(x, training=training))

        return x


def discriminator(inputs, training=True, reuse=False, alpha=0.2):
    """
    :param inputs: 训练时分别输入真实图像和generator的生成假图像
    :param training: BN层是否训练
    :param reuse: 是否重用变量
    :param alpha: leakly relu 的参数
    :return: 辨别器生成的logits，用tanh激活后的logits
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.conv2d(inputs, 64, 5, 2, 'same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha * x, x)

        x = tf.layers.conv2d(x, 128, 5, 2, 'same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha * x, x)

        x = tf.layers.conv2d(x, 256, 5, 2, 'same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha * x, x)

        x = tf.layers.conv2d(x, 512, 5, 2, 'same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha * x, x)

        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, 1)
        output = tf.nn.sigmoid(logits)

        return logits, output

#loss
def loss(input_z, input_real):
    """
    :param input_z: 随机生成的噪声
    :param input_real: 真实图像，输入前缩放到（-1,1）
    :return: d_loss: discriminator的损失值
              g_loss：generator的损失值
    """
    input_fake = generator(input_z)
    logits_real, output_real = discriminator(input_real)
    logits_fake, output_fake = discriminator(input_fake, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))

    d_loss = d_loss_fake + d_loss_real

    return d_loss, g_loss

#optimizers
def opts(d_loss, g_loss, learning_rate):
    """
    :param d_loss:
    :param g_loss:
    :param learning_rate: 学习率
    :return: 优化器
    """
    var_list = tf.trainable_variables()
    g_var = [var for var in var_list if var.name.startswith('generator')]
    d_var = [var for var in var_list if var.name.startswith('discriminator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_loss, var_list=g_var)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss, var_list=d_var)

    return g_opt, d_opt
