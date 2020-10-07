#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-7-19 上午10:28
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : test_model.py
# @IDE: PyCharm
"""
test model
"""
import os.path as ops
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

from attentive_gan_model import derain_drop_net
# from attentive_gan_model import derain_vgg_rcf
# from attentive_gan_model import derain_rcf_net
# from attentive_gan_model import derain_rcf_change
# from attentive_gan_model import derain_att_dense
# from attentive_gan_model import derain_att_dense_2
# from attentive_gan_model import derain_att_res_guide_non_local_dis

import os
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The input image path',default='../data/test_b/data/')

    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./att+psp_model/derain_gan_2019-11-10-13-06-06.ckpt-7000')
    parser.add_argument('--weights_path', type=str, help='The model weights path',default='../orig_model/derain_gan/derain_gan.ckpt-100000')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./model/att_vgg_rcf_gan/derain_gan_2019-11-15-10-09-42.ckpt-12000')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./model/att_vgg_rcf_gan_360_240/derain_gan_2019-11-16-16-23-29.ckpt-60500')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./model/att_rcf_gan_360_240/derain_gan_2019-11-17-12-03-15.ckpt-38500')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./model/att_rcf_change_360_240/derain_gan_2019-11-18-16-31-21.ckpt-300')
    # parser.add_argument('--weights_path', type=str, help='The model weights path', default='./model/att_dense_1_1/derain_gan_2019-11-29-10-44-08.ckpt-100000')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',default='./model/att_dense_2/derain_gan_2019-11-30-16-43-02.ckpt-100000')
    # parser.add_argument('--weights_path', type=str, help='The model weights path',
    #                     default='./model/att_res_non_local_dis/derain_gan_2019-12-10-15-32-03.ckpt-100000')

    parser.add_argument('--label_path', type=str, default='../data/test_b/gt/', help='The label image path')
    parser.add_argument('--output_path', type=str, default='../data/test_data/test_b_output/', help='The output image path')
    # parser.add_argument('--output_path', type=str, default='../data/test_data/ori_test_output/',help='The output image path')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_model(image_path, weights_path, label_path=None):
    """

    :param image_path:
    :param weights_path:
    :param label_path:
    :return:
    """
    assert ops.exists(image_path)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TEST.BATCH_SIZE, CFG.TEST.IMG_HEIGHT, CFG.TEST.IMG_WIDTH, 3],
                                  name='input_tensor'
                                  )



    phase = tf.constant('test', tf.string)

    net = derain_drop_net.DeRainNet(phase=phase)
    # net = derain_drop_pspnet.DeRainNet(phase=phase)
    # net = derain_cnn_psp.DeRainNet(phase=phase)
    # net = derain_rcf_change.DeRainNet(phase=phase)
    # net = derain_rcf_net.DeRainNet(phase=phase)
    # net = derain_att_dense.DeRainNet(phase=phase)
    # net = derain_att_dense_2.DeRainNet(phase=phase)
    # net = derain_att_res_guide_non_local_dis.DeRainNet(phase=phase)

    output, attention_maps,ret = net.inference(input_tensor=input_tensor, name='derain_net')
    # output, attention_maps = net.inference(input_tensor=input_tensor, name='derain_net')
    # Set sess configuration
    # sess_config = tf.ConfigProto(allow_soft_placement=False)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    # sess_config.gpu_options.allocator_type = 'BFC'

    # sess = tf.Session(config=sess_config)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=weights_path)
    with sess.as_default():


        im_names = os.listdir(image_path)
        num = len(im_names)
        lb_names = os.listdir(label_path)

        sum_psnr = 0
        sum_ssim = 0
        for i in range(num):
            image = cv2.imread(image_path+im_names[i], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            image_vis = image
            image = np.divide(np.array(image, np.float32), 127.5) - 1.0

            label_image_vis = None
            if label_path is not None:
                label_image = cv2.imread(label_path+ im_names[i].split('_')[0] + '_clean.jpg', cv2.IMREAD_COLOR)
                label_image_vis = cv2.resize(
                    label_image, (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
                )

            output_image, atte_maps = sess.run(
                [output, attention_maps],
                feed_dict={input_tensor: np.expand_dims(image, 0)})

            output_image = output_image[0]
            for j in range(output_image.shape[2]):
                output_image[:, :, j] = minmax_scale(output_image[:, :, j])

            output_image = np.array(output_image, np.uint8)

            # 保存并可视化结果
            # cv2.imwrite(args.output_path + im_names[i], image_vis)
            cv2.imwrite(args.output_path + im_names[i].split('_')[0] + '_derain.png', output_image)

            # plt.figure('src_image')
            # plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.figure('derain_ret')
            plt.imshow(output_image[:, :, (2, 1, 0)])
            # plt.figure('atte_map_1')
            # plt.imshow(atte_maps[0][0, :, :, 0], cmap='jet')
            # plt.savefig(args.output_path + str(i) + 'atte_map_1.png')
            # plt.figure('atte_map_2')
            # plt.imshow(atte_maps[1][0, :, :, 0], cmap='jet')
            # plt.savefig(args.output_path + str(i) + 'atte_map_2.png')
            # plt.figure('atte_map_3')
            # plt.imshow(atte_maps[2][0, :, :, 0], cmap='jet')
            # plt.savefig(args.output_path + str(i) + 'atte_map_3.png')
            # plt.figure('atte_map_4')
            # plt.imshow(atte_maps[3][0, :, :, 0], cmap='jet')
            # plt.savefig(args.output_path + str(i) + 'atte_map_4.png')
            plt.show()

            if label_path is not None:
                label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
                psnr = compare_psnr(label_image_vis_gray, output_image_gray)
                ssim = compare_ssim(label_image_vis_gray, output_image_gray)

                sum_psnr = sum_psnr + psnr
                sum_ssim = sum_ssim + ssim
                print(i)
                print('SSIM: {:.5f}'.format(ssim))
                print('PSNR: {:.5f}'.format(psnr))



        avg_ssim = sum_ssim/num
        avg_psnr = sum_psnr/num
        print('avg_SSIM: {:.5f}'.format(avg_ssim))
        print('avg_PSNR: {:.5f}'.format(avg_psnr))
    return


if __name__ == '__main__':
    import datetime
    starttime = datetime.datetime.now()
    print("当前时间: ", str(starttime).split('.')[0])
    # init args
    args = init_args()

    # test model
    with tf.device('/cpu:0'):
        test_model(args.image_path, args.weights_path, args.label_path)
    endtime = datetime.datetime.now()
    print("结束时间: ", str(endtime).split('.')[0])
    print(u'相差：%s' % (endtime - starttime))
