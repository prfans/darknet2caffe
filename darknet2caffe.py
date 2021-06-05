# coding: utf-8

import os
import sys
import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *


def darknet2caffe(cfgfile, weightfile, protofile, caffemodel, save_cfg_blocks=True, save_caffenet_params=True):
    # 根据prototxt文件创建caffe网络
    net = caffe.Net(protofile, caffe.TEST)

    # 获取所有的caffe网络参数
    params = net.params
    if save_caffenet_params:
        with open("./caffe_net_params.txt", "w") as fp:
            for key in params:
                fp.write("{}: ".format(key))
                for i in range(len(params[key])):
                    shape_str = "("
                    for item in params[key][i].data.shape:
                        shape_str += str(item) + ","
                    shape_str += ")  "
                    fp.write(shape_str)
                fp.write("\n")

    # 打开weights文件并加载
    print('Opening weights: {} ...... '.format(weightfile))
    with open(weightfile, "rb") as fp:
        # 读取版本等信息
        major_version = np.fromfile(fp, dtype=np.int32, count=1)
        minor_version = np.fromfile(fp, dtype=np.int32, count=1)
        revision = np.fromfile(fp, dtype=np.int32, count=1)
        images_seen = np.fromfile(fp, dtype=np.uint64, count=1)
        print("     major: ", major_version[0])
        print("     minor: ", minor_version[0])
        print("     revision: ", revision[0])
        print("     images seen: ", images_seen[0])

        # 读取余下的权重数据
        buf = np.fromfile(fp, dtype=np.float32)
        print("     weights shape: ", buf.shape)

    # 解析darknet cfg网络描述文件
    print('Parsing cfgfile: {} ......'.format(cfgfile))
    blocks = parse_cfg(cfgfile)
    if save_cfg_blocks:
        with open('./cfg_blocks.txt', 'w') as fp:
            block_layers = 0
            for block in blocks:
                fp.write("---------------{}-------------\n".format(block_layers))
                saved_layer_str = ""
                for key in block:
                    saved_layer_str += key + ":" + str(block[key]) + ", "
                fp.write(saved_layer_str)
                fp.write("\n")
                block_layers += 1

    # 按照顺序把darknet weights数据填充到caffe params位置
    print('Moving weights >>>>>> caffemodel ......')
    layers = []
    layer_id = 1
    start = 0
    for block in blocks:
        if block['type'] == 'net':
            continue

        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            if 'name' in block:
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
                scale_layer_name = '%s-scale' % block['name']
            else:
                conv_layer_name = 'layer%d-conv' % layer_id
                bn_layer_name = 'layer%d-bn' % layer_id
                scale_layer_name = 'layer%d-scale' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start,
                                           params[conv_layer_name],
                                           params[bn_layer_name],
                                           params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start,
                                        params[conv_layer_name])
            layer_id = layer_id+1

        elif block['type'] == 'depthwise_convolutional':
            batch_normalize = int(block['batch_normalize'])
            if 'name' in block:
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
                scale_layer_name = '%s-scale' % block['name']
            else:
                conv_layer_name = 'layer%d-dwconv' % layer_id
                bn_layer_name = 'layer%d-bn' % layer_id
                scale_layer_name = 'layer%d-scale' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start,
                                           params[conv_layer_name],
                                           params[bn_layer_name],
                                           params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start,
                                        params[conv_layer_name])
            layer_id = layer_id+1

        elif block['type'] == 'upsample':

            # 重写反卷积层的参数
            print('     layer: %d upsample, set weights = 1.0f, bias = 0.0f' % layer_id)
            if 'name' in block:
                layer_name = block['name']
            else:
                layer_name = 'layer%d-upsample' % layer_id
            params[layer_name][0].data[...] = np.array([1.0])
            try:
                params[layer_name][1].data[...] = np.array([0.0])
            except:
                pass
            layer_id = layer_id + 1

        elif block['type'] == 'connected':
            if 'name' in block:
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d-fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id+1

        elif block['type'] == 'maxpool':
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            layer_id = layer_id+1
        elif block['type'] == 'region':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1
        elif block['type'] == 'cost':
            layer_id = layer_id + 1
        else:
            print('     layer: %d, unknow layer type %s ' % (layer_id, block['type']))
            layer_id = layer_id + 1
        if start >= buf.size:
            print(' Warn: read end of weights buf.....')
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)


def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + (bias.size)
    conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + (weight.size)
    return start


def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start


def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
    conv_weight = conv_param[0].data
    running_mean = bn_param[0].data
    running_var = bn_param[1].data
    scale_weight = scale_param[0].data
    scale_bias = scale_param[1].data

    scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size

    bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    bn_param[2].data[...] = np.array([1.0])

    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + (conv_weight.size)

    return start


def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    prev_filters = 3
    layers = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 1
    topnames = dict()
    for block in blocks:
        if block['type'] == 'net':
            props['name'] = 'Darkent2Caffe'
            props['input'] = 'data'
            props['input_dim'] = ['1']
            props['input_dim'].append(block['channels'])
            props['input_dim'].append(block['height'])
            props['input_dim'].append(block['width'])
            continue

        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            conv_layer['bottom'] = bottom
            if 'name' in block:
                conv_layer['top'] = block['name']
                conv_layer['name'] = block['name']
            else:
                conv_layer['top'] = 'layer%d-conv' % layer_id
                conv_layer['name'] = 'layer%d-conv' % layer_id
            conv_layer['type'] = 'Convolution'
            convolution_param = OrderedDict()
            convolution_param['num_output'] = block['filters']
            prev_filters = block['filters']
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size']) // 2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'

            # 分组卷积参数
            if 'groups' in block:
                convolution_param['group'] = block['groups']

            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                if 'name' in block:
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                scale_layer['bottom'] = bottom
                scale_layer['top'] = bottom
                if 'name' in block:
                    scale_layer['name'] = '%s-scale' % block['name']
                else:
                    scale_layer['name'] = 'layer%d-scale' % layer_id
                scale_layer['type'] = 'Scale'
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                activate_layer = OrderedDict()
                activate_layer['bottom'] = bottom
                activate_layer['top'] = bottom
                if 'name' in block:
                    activate_layer['name'] = '%s-act' % block['name']
                else:
                    activate_layer['name'] = 'layer%d-act' % layer_id
                if block['activation'] == 'leaky':
                    activate_layer['type'] = 'ReLU'
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    activate_layer['relu_param'] = relu_param
                elif block['activation'] == 'mish':
                    activate_layer['type'] = 'Mish' 
                layers.append(activate_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1

        elif block['type'] == 'depthwise_convolutional':
            conv_layer = OrderedDict()
            conv_layer['bottom'] = bottom
            if 'name' in block:
                conv_layer['top'] = block['name']
                conv_layer['name'] = block['name']
            else:
                conv_layer['top'] = 'layer%d-dwconv' % layer_id
                conv_layer['name'] = 'layer%d-dwconv' % layer_id
            conv_layer['type'] = 'ConvolutionDepthwise'
            convolution_param = OrderedDict()
            convolution_param['num_output'] = prev_filters
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size']) // 2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                if 'name' in block:
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                scale_layer['bottom'] = bottom
                scale_layer['top'] = bottom
                if 'name' in block:
                    scale_layer['name'] = '%s-scale' % block['name']
                else:
                    scale_layer['name'] = 'layer%d-scale' % layer_id
                scale_layer['type'] = 'Scale'
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1

        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            max_layer['bottom'] = bottom
            if 'name' in block:
                max_layer['top'] = block['name']
                max_layer['name'] = block['name']
            else:
                max_layer['top'] = 'layer%d-maxpool' % layer_id
                max_layer['name'] = 'layer%d-maxpool' % layer_id
            max_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['stride'] = block['stride']
            pooling_param['pool'] = 'MAX'
            # pooling_param['kernel_size'] = block['size']
            # pooling_param['pad'] = str((int(block['size'])-1) // 2)
            if (int(block['size']) - int(block['stride'])) % 2 == 0:
                pooling_param['kernel_size'] = block['size']
                pooling_param['pad'] = str((int(block['size'])-1) // 2)

            if (int(block['size']) - int(block['stride'])) % 2 == 1:
                pooling_param['kernel_size'] = str(int(block['size']) + 1)
                pooling_param['pad'] = str((int(block['size']) + 1) // 2)
            
            max_layer['pooling_param'] = pooling_param
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1

        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            avg_layer['bottom'] = bottom
            if 'name' in block:
                avg_layer['top'] = block['name']
                avg_layer['name'] = block['name']
            else:
                avg_layer['top'] = 'layer%d-avgpool' % layer_id
                avg_layer['name'] = 'layer%d-avgpool' % layer_id
            avg_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = 7
            pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1

        elif block['type'] == 'region':
            if True:
                region_layer = OrderedDict()
                region_layer['bottom'] = bottom
                if 'name' in block:
                    region_layer['top'] = block['name']
                    region_layer['name'] = block['name']
                else:
                    region_layer['top'] = 'layer%d-region' % layer_id
                    region_layer['name'] = 'layer%d-region' % layer_id
                region_layer['type'] = 'Region'
                region_param = OrderedDict()
                region_param['anchors'] = block['anchors'].strip()
                region_param['classes'] = block['classes']
                region_param['num'] = block['num']
                region_layer['region_param'] = region_param
                layers.append(region_layer)
                bottom = region_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

        elif block['type'] == 'route':
            route_layer = OrderedDict()
            layer_name = str(block['layers']).split(',')
            bottom_layer_size = len(str(block['layers']).split(','))
            bottoms = []
            for i in range(bottom_layer_size):
                if int(layer_name[i]) < 0:
                    prev_layer_id = layer_id + int(layer_name[i])
                else:
                    prev_layer_id = int(layer_name[i]) + 1
                bottom = topnames[prev_layer_id]
                bottoms.append(bottom)

            route_layer['bottom'] = bottoms

            if 'name' in block:
                route_layer['top'] = block['name']
                route_layer['name'] = block['name']
            else:
                route_layer['top'] = 'layer%d-route' % layer_id
                route_layer['name'] = 'layer%d-route' % layer_id
            route_layer['type'] = 'Concat'
            layers.append(route_layer)
            bottom = route_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

        elif block['type'] == 'upsample':
            upsample_layer = OrderedDict()
            upsample_layer['bottom'] = bottom
            if 'name' in block:
                upsample_layer['top'] = block['name']
                upsample_layer['name'] = block['name']
            else:
                upsample_layer['top'] = 'layer%d-upsample' % layer_id
                upsample_layer['name'] = 'layer%d-upsample' % layer_id
            upsample_layer['type'] = 'Upsample'
            upsample_param = OrderedDict()
            upsample_param['scale'] = block['stride']
            upsample_layer['upsample_param'] = upsample_param
            layers.append(upsample_layer)
            bottom = upsample_layer['top']
            print(' upsample:', layer_id)
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            shortcut_layer = OrderedDict()
            shortcut_layer['bottom'] = [bottom1, bottom2]
            if 'name' in block:
                shortcut_layer['top'] = block['name']
                shortcut_layer['name'] = block['name']
            else:
                shortcut_layer['top'] = 'layer%d-shortcut' % layer_id
                shortcut_layer['name'] = 'layer%d-shortcut' % layer_id
            shortcut_layer['type'] = 'Eltwise'
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id + 1           
            
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            fc_layer['bottom'] = bottom
            if 'name' in block:
                fc_layer['top'] = block['name']
                fc_layer['name'] = block['name']
            else:
                fc_layer['top'] = 'layer%d-fc' % layer_id
                fc_layer['name'] = 'layer%d-fc' % layer_id
            fc_layer['type'] = 'InnerProduct'
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        else:
            print(' layer: %d, unknow layer type %s ' % (layer_id, block['type']))
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('python darknet2caffe.py xxx.cfg xxx.weights xxx.prototxt xxx.caffemodel')
        print('please add name field for each block to avoid generated name')
        exit()

    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    protofile = sys.argv[3]
    caffemodel = sys.argv[4]
    if os.path.exists(protofile):
        darknet2caffe(cfgfile, weightfile, protofile, caffemodel)
    else:
        print('generating prototxt ......')
        net_info = cfg2prototxt(cfgfile)
        save_prototxt(net_info, protofile)
        print('save prototxt to %s, please re-try' % protofile)
