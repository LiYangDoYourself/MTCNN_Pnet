# coding:utf-8

import tensorflow as tf
import os, sys
import random

"""
1 定义tensorflow的读取文件的结构
2 读取数据写入tfrecord格式
"""


class ImageConverToor(object):

    def __init__(self):
        # 初始化的时候就得开启一个会话
        self._sess = tf.Session()

        # 定义一个站位符
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=1)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format="gray", quality=100)

        #
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3, "jpeg needs height width channel"
        assert image.shape[2] == 1
        return image


coder = ImageConverToor()


def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _is_png(filename):
    return filename.endswith(".png")


def _process_image_withcoder(filename):
    with tf.gfile.FastGFile(filename) as f:
        image = f.read()

    if _is_png(filename):
        print("将{} png转换成jpeg".format(filename))
        image = coder.png_to_jpeg(image)
    else:
        print("将{} 解码成jpeg".format(filename))
        image = coder.decode_jpeg(image)

    image = image.tostring()

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 1
    return image, height, width


def _convert_to_example_simple(example_data, image_data):
    # 先获取到对应的数据和标签
    class_label = example_data["label"]
    bbox = example_data["bbox"]
    roi = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]

    # 得到对象 要固话进去的数据
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': _bytes_feature(image_data),
            'image/label': _int64_feature(class_label),
            'image/roi': _float_feature(roi)
        }
    ))

    return example


def _add_to_tfrecord(filename, example_data, write_tfrecord):
    image_data, height, width = _process_image_withcoder(filename)
    example = _convert_to_example_simple(example_data, image_data)
    write_tfrecord.write(example.SerializeToString())


def get_dataset(read_txt):
    all_dataset = []

    with open(read_txt) as f:
        dataset = f.readlines()

        for per_data in dataset:
            per_dict = {}
            bbox_dict = {}
            data_list = per_data.split(" ").strip("\n\r")
            per_dict["filename"] = data_list[0]
            per_dict["label"] = data_list[1]

            bbox_dict["xmin"] = data_list[2]
            bbox_dict["ymin"] = data_list[3]
            bbox_dict["xmax"] = data_list[4]
            bbox_dict["ymax"] = data_list[5]

            per_dict["bbox"] = bbox_dict
        all_dataset.append(per_dict)
    return all_dataset


def run(main_dir):
    gen_tfrecord = os.path.join(main_path, 'pnet_no_mark.tfrecord')
    if tf.gfile.Exists(gen_tfrecord):
        print("该文件已经存在 ")
        return

    read_txt = os.path.join(main_dir, 'a.txt')

    # 一个列表
    dataset = get_dataset(read_txt)
    # 随机打算列表
    random.shuffle(dataset)

    # 开启tensorflow 文件写入器
    with tf.python_io.TFRecordWriter(gen_tfrecord) as write_tfrecord:

        for i, example_data in enumerate(dataset):
            if (i + 1) % 100 == 0:
                sys.stdout.write('已经转换了{}/{}'.format(i, len(dataset)))
            sys.stdout.flush()
            filename = example_data["filename"]
            _add_to_tfrecord(filename, example_data, write_tfrecord)

    print("\n数据转换成功-------------------!")

    pass


if __name__ == '__main__':
    main_path = "../../DATA/imglist"

    run(main_path)
