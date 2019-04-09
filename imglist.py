# coding:utf-8
import os
import numpy.random as npr

main_path = "../../DATA/10/imglist/a.txt"
pos_path = "../../DATA/10/pos_10.txt"
neg_path = "../../DATD/10/neg_10.txt"
part_path = "../../DATA/10/part_10.txt"

# f = open(main_path, 'wb')


with open(pos_path) as f1:
    pos_data = f1.readlines()

with open(neg_path) as f2:
    neg_data = f2.readlines()

with open(part_path) as f3:
    part_data = f3.readlines()

print "pos:{},neg:{},part:{}".format(len(pos_data), len(neg_data), len(part_data))

with open(main_path, 'wb') as f:
    radio = [2, 1, 1]

    base_num = len(pos_data)

    pos_num = npr.choice(len(pos_data), size=len(pos_data), replace=False)
    if radio[0] * base_num > len(neg_data):
        neg_num = npr.choice(len(neg_data), size=len(neg_data), replace=False)
    else:
        neg_num = npr.choice(len(neg_data), size=radio[0] * base_num, replace=False)
    part_num = npr.choice(len(part_data), size=radio[1] * base_num, replace=False)

    for i in pos_num:
        f.write(pos_data[i])
    for j in part_num:
        f.write(part_data[j])
    for k in neg_num:
        f.write(neg_data[k])
