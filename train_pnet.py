# coding:utf-8
import tensorflow as tf
import random, os
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector


def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)

    # 将部分样本标签值为0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)

    # 统计分类的图片个数
    num_cls_prob = tf.size(cls_prob)

    # 分类 自动计算多少类 num_cls_prob 多列
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])

    # 转换类型
    label_int = tf.cast(label_filter_invalid, tf.int32)

    if not cls_prob.get_shape().as_list()[0]:
        num_row = tf.constant(config.BATCH_SIZE)
    else:
        num_row = tf.to_int32(cls_prob.get_shape()[0])

    row = tf.range(num_row) * 2
    indices_ = row + label_int

    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)

    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros, zeros, ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)

    return tf.reduce_mean(loss)
    pass


def bbox_ohem(bbox_pred, bbox_target, label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)

    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)

    square_error = tf.squeeze(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)

    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds


def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))

    # 将所有小于0值为0
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def _activation_summary(x):
    tensor_name = x.op.name
    print('load summary for:', tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x)


def P_Net(inputs, label=None, bbox_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print(inputs.get_shape())

        # 开始卷积
        net = slim.conv2d(inputs, num_outputs=10, kernel_size=[3, 3], stride=1, scope='conv1')
        _activation_summary(net)
        print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
        _activation_summary(net)
        print(net.get_shape())

        net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
        _activation_summary(net)
        print(net.get_shape())

        net = slim.conv2d(net, num_outputs=32, kernel_size=[2, 2], stride=1, scope='conv3')
        _activation_summary(net)
        print(net.get_shape())

        conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 4], stride=1, scope='conv4_1',
                              activation_fn=tf.nn.softmax)
        _activation_summary(conv4_1)
        print(conv4_1.get_shape())

        bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 4], stride=1, scope='conv4_2', activation_fn=None)
        _activation_summary(bbox_pred)
        print(bbox_pred.get_shape())

    if training:
        cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')

        cls_loss = cls_ohem(cls_prob, label)

        bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')

        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)

        accuracy = cal_accuracy(cls_prob, label)
        L2_loss = tf.add_n(slim.losses.get_regularization_losses())
        # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        return cls_loss, bbox_loss, L2_loss, accuracy


    else:
        cls_pro_test = tf.squeeze(conv4_1, axis=0)
        bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
        # landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
        # return cls_pro_test,bbox_pred_test,landmark_pred_test
        return cls_pro_test, bbox_pred_test


def read_single_tfrecord(dataset_dir, batch_size, net, proportion):
    # 输出字符串到一个输入管道队列

    filename_queue = tf.train.string_input_producer([dataset_dir], shuffle=True)

    # 开启tf的读取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解析器
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32)
        }
    )

    if net == 'PNet':
        image_size = 10

    elif net == 'RNet':
        image_size = 20
    else:
        image_size = 40

    # 先解析
    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    # 然后调整大小到我们指定大小
    image = tf.reshape(image, [image_size, int(proportion * image_size), 1])
    # 随机调整饱和度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机调整亮度
    tf.image.random_brightness(image, max_delta=0.2)
    # 添加到tensorboard中去
    tf.summary.image('%s' % random.randint(0, 1000000), tf.expand_dims(image, 0))
    # 归一化
    image = (tf.cast(image, tf.float32) - 127.5) / 128

    # 然后取出标签
    label = tf.cast(image_features['image/label'], tf.float32)
    # bbox
    roi = tf.cast(image_features['image/roi'], tf.float32)

    # 获取batchsize的数据
    image, label, roi = tf.train.batch(
        [image, label, roi],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )

    label = tf.reshape(label, [batch_size])
    roi = tf.reshape(roi, [batch_size, 4])

    return image, label, roi


def train_model(base_lr, loss, data_num):
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / 64) for epoch in [6, 14, 20]]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, 4)]

    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)

    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)

    return train_op, lr_op


def train(net_factory, prefix, end_epoch, base_dir, proportion, display=display, base_lr=lr):
    net = prefix.split('/')[-1]

    label_file = os.path.join(base_dir, 'a.txt')

    print(label_file)

    f = open(label_file, 'r')
    # 多少张图片
    num = len(f.readlines())
    print("Total size of the dataset is :", num)
    print(prefix)

    if net == 'PNet':
        dataset_dir = os.path.join(base_dir, 'train_no_landmark.tfrecord')
        print('dataset dir is:', dataset_dir)
        image_batch.label_batch, bbox_batch = read_single_tfrecord(dataset_dir, 64, net, proportion)

    if net == 'PNet':
        image_size = 10
        radio_cls_loss = 1
        radio_bbox_loss = 0.5

    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_image')

    label = tf.placeholder(tf.float32, shape=[None], name='label')

    bbox_target = tf.placeholder(tf.float32, shape=[None, 4], name='bbox_target')

    cls_loss_op, bbox_loss_op, L2_loss_op, accuracy_op = net_factory(input_image, label, bbox_target, training=True)

    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + L2_loss_op

    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num)

    # 初始化一下全局变量防止出错
    init = tf.global_variables_initializer()
    sess = tf.Session()

    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    # 将这些损失添加到summary中
    tf.summary.scalar("cls_loss", cls_loss_op)
    tf.summary.scalar("bbox_loss", bbox_loss_op)
    tf.summary.scalar("cls_accuarcy", accuracy_op)
    tf.summary.scalar("total_loss", total_loss_op)

    # 最后合并summary
    summary_op = tf.summary.merge_all()

    # 日志文件
    logs_dir = "../logs/%s4" % net

    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)

    # 把上面写入图表中
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)

    # 建一个线程管理器（协调器）对象
    coord = tf.train.Coordinator()

    # 开启入线程
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    MAX_STEP = int(num / 64) * end_epoch
    epoch = 0

    sess.graph.finalize()

    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break

            image_batch_array, label_batch_array, bbox_batch_array = sess.run([image_batch, label_batch, bbox_batch])
            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array, label: label_batch_array,
                                                bbox_target: bbox_batch_array})

            if (step + 1) % display == 0:
                cls_loss, bbox_loss, l2_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, L2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array}
                )
                total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss

                print(
                        "%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                    datetime.now(), step + 1, MAX_STEP, acc, cls_loss, bbox_loss, L2_loss, total_loss, lr))
            if i * 64 > num * 2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch * 2)
                print('path prefix is:', path_prefix)

            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成")

    finally:
        coord.request_stop()
        writer.close()
    coord.join(thread)
    sess.close()


def train_PNet():
    net_factory = P_Net
    train(net_factory, prefix, end_epoch, base_dir, proportion, display=display, base_lr=lr)

    pass


if __name__ == '__main__':
    base_dir = '../../DATA/imglist/'
    model_name = 'MTCNN'
    model_path = "../data/%s_model/Pnet_no_landmark4/PNet"

    prefix = model_path

    end_epoch = 100
    display = 100
    lr = 0.001
    proportion = 1.5

    train_PNet(base_dir, prefix, end_epoch, display, lr, proportion)
