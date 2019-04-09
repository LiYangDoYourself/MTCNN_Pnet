# coding:utf-8
"""
1 我们已经有了tran_xml.txt文件 内容:图片的路径 2018-10-30-14-58-36/0661.jpeg 144 110 155 268
2 先创建 主文件（10）-分类（positive,negative,part）-positive(剪切的图片的12×12,iou比要大于0.8)
                                                -negative(12*12 0.3)
                                               -part(12*12 0.6-0.65)
"""
import os
import numpy as np
import numpy.random as npr
import cv2

main_path = "../../DATA20"
size = 10
read_txt = "prepare_data/tran_xml.txt"
cls_name = ["positive", "negative", "part"]
restore_path = "../DATA20/WIDER_train/images"
proportion = 1.5  # w/h

# 记录读取图片的数量，pos的数量,neg的数量
pos_id = 0
neg_id = 0
part_id = 0
img_id = 0
box_id = 0

# 创建对应目录
for name in cls_name:
    cls_path = os.path.join(main_path, size, name)
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)

f1 = open(os.path.join(main_path, size, "pos_{}.txt".format(size)))
f2 = open(os.path.join(main_path, size, "neg_{}.txt".format(size)))
f3 = open(os.path.join(main_path, size, "part_{}.txt".format(size)))


def IOU(crop_bbox, mark_bbox):
    crop_area = (crop_box[2] - crop_box[0] + 1) * (crop_box[3] - crop_box[1] + 1)
    mark_area = (mark_bbox[:, 2] - mark_bbox[:, 0] + 1) * (mark_bbox[:, 3] - mark_bbox[:, 1] + 1)

    xx1 = np.maximum(crop_box[0], mark_box[:, 0])
    yy1 = np.maximum(crop_box[1], mark_bbox[:, 1])
    xx2 = np.minimum(crop_box[2], mark_bbox[:, 2])
    yy2 = np.minimum(crop_box[3], mark_bbox[:, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h

    iou = inter / (crop_area + mark_area - inter)

    return iou


# 读取图片
with open(read_txt, 'r') as f:
    annotion_data = f.readlines()

# 一共有多少张图片
print("pic num :%d"%len(annotion_data))
# 循环读取图片的路径
for annotion_path in annotion_data:
    anno_list = annotion_path.split(' ').strip("\n\r")
    img_path = anno_list[0]  # 不是全路径 得拼接
    mark_box = np.array(anno_list[1:]).reshape(-1, 4)
    try:
        img = cv2.imread(os.path.join(restore_path, img_path))
        img_id += 1
        height, width, channel = img.shape
    except Exception as e:
        print(e)

    # 生成负样本 循环5次产生 在每张图片上产生5个负样本
    # 随机产生一个size 范围大小剪切的图片大小-最小的长宽的一半

    neg_num = 0
    while neg_num < 5:
        # 随机产生长宽 （范围最小10,读取图片长宽的一半）
        size = npr.randint(size, min(height, width) / 2)
        size_w = size * proportion

        # 随机产生一个初始点 0,width-size_w, 0,height-size
        nx = npr.randint(0, width - size_w)
        ny = npr.randint(0, height - size)

        # 截取图片
        crop_img = img[ny:ny + size, nx:nx + size_w, :]
        # 将四点坐标添加到 数组中
        crop_box = np.array([nx, ny, nx + size_w, ny + size])
        # 剪切resize
        resize_img = cv2.resize(crop_img, (size * proportion, size), interpolation=cv2.INTER_LINEAR)
        # 然后做IOU得出得分最大的
        iou = IOU(crop_box, mark_box)
        if np.max(iou) < 0.3:
            wirte_name = os.path.join(main_path, size, "negative/{}.jpg".format(neg_id))
            cv2.imwrite(wirte_name, resize_img)
            f2.write(wirte_name + " 0\n")
            neg_num += 1
            neg_id += 1

    # 在标记框周围产生负样本
    for bbox in mark_box:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if max(w, h) < 20 or x1 < 0 or x2 < 0:
            continue
        for i in range(5):
            size = npr.randint(size, min(width, height))
            size_w = size * proportion

            # 在x1,y1周围随机产生点 范围 max(-size_w,-x1),w
            delta_x = npr.randint(max(-size_w, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)

            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            nx2 = nx1 + size_w
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue

            # 剪切图片
            crop_img = img[ny1:ny2, nx1:nx2, :]

            # crop的边框
            crop_box = np.array([nx1, ny1, nx2, ny2])

            # 剪切出图片
            resize_img = cv2.resize(crop_img, (size * proportion, size), interpolation=cv2.INTER_LINEAR)

            iou = IOU(crop_box, bbox)

            if iou < 0.3:
                wirte_name = os.path.join(main_path, size, "negative/{}.jpg".format(neg_id))
                cv2.imwrite(wirte_name, resize_img)
                f2.write(wirte_name + " 0\n")
                neg_id += 1

        for i in range(20):
            size = npr.randint(0.8 * min(h, w), 1.25 * max(h, w))
            size_w = size * proportion

            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(0, x1 + w / 2 + delta_x - size_w / 2))
            ny1 = int(max(0, y1 + h / 2 + delta_y - size / 2))

            nx2 = nx1 + size_w
            ny2 = ny1 + size

            offset_x1 = (x1 - nx1) / size_w
            offset_y1 = (y1 - ny1) / size
            offset_x2 = (x2 - nx2) / size_w
            offset_y2 = (y2 - ny2) / size

            crop_img = img[ny1:ny2, nx1:nx2, :]
            resize_img = cv2.resize(crop_img, (size * proportion, size), interpolation=cv2.INTER_LINEAR)
            crop_bbox = np.array([nx1, ny1, nx2, ny2])

            iou = IOU(crop_box, bbox)
            if iou >= 0.8:
                write_name = os.path.join(main_path, size, "positive/{}.jpg".format(pos_id))
                f1.write(wirte_name + " 1 {} {} {} {}\n".format(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(write_name, resize_img)
                pos_id += 1

            elif 0.6 <= iou <= 0.65:
                write_name = os.path.join(main_path, size, "part/{}.jpg".format(part_id))
                f1.write(wirte_name + " -1 {} {} {} {}\n".format(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(write_name, resize_img)
                part_id += 1

        box_id += 1
    img_id += 1

f1.close()
f2.close()
f3.close()
