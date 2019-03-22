import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

'''
CBAM：convolution block attention module的主要思想是，首先在H/W维度做最大和平均池化，然后concat做一次全连接（控制全连接层输出通道的比例，一般为0.5）
激活函数为relu，然后再进行一次全连接，输出层通道数为原始输入feature map的通道数，激活函数为relu。输出结果与原始输入feature map进行数乘。数乘结果与原始输入的shape完全一致。
再将这个中间结果在C维度上进行reduce_mean和reduce_max，然后concat，做卷积操作，输出通道为1，激活函数为sigmoid，然后将其与中间结果数乘，即为最终结果。输出结果与输入的shape完全一致。
'''


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention

        # 在H/W上做平均和最大池化 池化核大小为feature map的H/W
        channel_avg_weights = tf.nn.avg_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_max_weights = tf.nn.max_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )

        # NHWC形状变为N*1*1*C，reshape为N*1*C
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # 进行concat，形状变为N*2*C
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        # 进行一个全连接层操作,输出维度通过inner_units_ratio来控制，即输入通道数的比例，如0.5，那么结果的维度为N*2*0.5C,激活函数为relu
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        # 进行一个全连接层操作,输入维度为N*2*0.5C,输出维度为N*2*C,激活函数为sigmoid
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=tf.nn.sigmoid
        )
        # 输入维度N*2*C，在index=1的维度上进行求和，形状为N*1*1*C
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        # 与原始输入的feature map数乘，即对应位置相乘[2,1,1,32],[2,8,8,32]-->[2,8,8,32]
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        # spatial attention 在通道的维度上进行均值和最大池化 即N*H*W*C---->N*H*W*1
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])

        # 将均值和最大池化的通道维度上进行concat，维度变为N*H*W*2
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 进行一次卷积，输出通道为1，那么结果为N*H*W*1,激活函数为sigmoid
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [3, 3],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 与原始输入的feature_map_with_channel_attention数乘，即对应位置相乘[2,8,8,1],[2,8,8,32]-->[2,8,8,32]
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention


if __name__ == '__main__':
    # example
    feature_map = tf.constant(np.random.rand(2, 8, 8, 32), dtype=tf.float16)
    feature_map_with_attention = convolutional_block_attention_module(feature_map, 0)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        result = sess.run(feature_map_with_attention)
        print(result.shape)
