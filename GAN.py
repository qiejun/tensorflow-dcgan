from utils import *
from data import *
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self):
        tf.reset_default_graph()
        self.input_z = tf.placeholder(tf.float32, [None, 100])
        self.input_real = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.d_loss, self.g_loss = loss(input_z=self.input_z, input_real=self.input_real)
        self.d_opt, self.g_opt = opts(self.d_loss, self.g_loss, learning_rate=0.0002)

    def view_samples(self, imgs_list, rows=2, cols=5):
        fig = plt.figure(figsize=(5, 2))
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, (i + 1))
            ax.axis('off')
            img = (imgs_list[i] + 1) / 2
            ax.imshow(img)
        plt.show()

    def train(self, epoches=100, batch_size=64):
        saver = tf.train.Saver()
        sample_z = np.random.uniform(-1, 1, size=(batch_size, 100))
        steps = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epoches):
                X = load_data('E:\\GAN_pictures', batch_size=batch_size)
                batch_z = np.random.uniform(-1, 1, [batch_size, 100])
                for x_real in X:
                    steps = steps + 1
                    _ = sess.run(self.d_opt, feed_dict={self.input_z: batch_z, self.input_real: x_real})
                    _ = sess.run(self.g_opt, feed_dict={self.input_z: batch_z, self.input_real: x_real})
                    if steps % 10 == 0:
                        train_loss_d = sess.run(self.d_loss, feed_dict={self.input_z: batch_z, self.input_real: x_real})
                        train_loss_g = sess.run(self.g_loss, feed_dict={self.input_z: batch_z, self.input_real: x_real})
                        print('epoch:', e, ',steps:', steps, ',d_loss:', train_loss_d, ',g_loss:', train_loss_g)
                    if steps % 100 == 0:
                        samples_list = []
                        g_samples = sess.run(generator(inputs_z=self.input_z, reuse=True, training=False),
                                             feed_dict={self.input_z: sample_z})
                        for i in range(batch_size):
                            samples_list.append(g_samples[i, ...])
                        self.view_samples(samples_list, 2, 5)
                if e >= 20 and e % 5 == 0:
                    saver.save(sess=sess, save_path='save/save.ckpt' + str(e))


if __name__ == '__mian()__':
    gan = DCGAN()
    gan.train()
