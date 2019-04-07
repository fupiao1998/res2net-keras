from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import add


def Conv_bn_relu(num_filters,
                 kernel_size,
                 batchnorm=True,
                 strides=(1, 1),
                 padding='same'):

    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer


def SEblock():
    def layer(input_tensor):

        return x
    return layer

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list


def res2net_block(num_filters, slice_num):
    def layer(input_tensor):
        short_cut = input_tensor
        x = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(input_tensor)
        slice_list = slice_layer(x, slice_num, x.shape[-1])
        side = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
        z = concatenate([slice_list[0], side])   # for one and second stage
        for i in range(2, len(slice_list)):
            y = Conv_bn_relu(num_filters=num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
            side = y
            z = concatenate([z, y])
        z = Conv_bn_relu(num_filters=num_filters, kernel_size=(1, 1))(z)
        out = concatenate([z, short_cut])
        return out
    return layer


x = Input((256, 256, 256))
print(x.shape)
x_conv_nor = Conv_bn_relu(512, (3, 3))(x)
print(x_conv_nor.shape)
out = slice_layer(x_conv_nor, 8, 512)
print(out)
print(len(out))
x = res2net_block(512, 8)(x_conv_nor)
print(x.shape)