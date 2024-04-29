# DiffSinger-backbones-mod

本repo存放了多个可以使用在[OpenVPI/Diffsinger](https://github.com/openvpi/DiffSinger)的backbones网络

将本repo文件替换OpenVPI/Diffsinger内文件，修改config中`backbone_type`即可使用

## lynxnet

出自：https://github.com/CNChTu/FCPE

本repo参考：https://github.com/yxlllc/ReFlow-VAE-SVC/blob/main/reflow/naive_v2_diff.py

训练和推理的过程中，有更快的速度以及更小的显存占用

经测试在数据音域边缘的表现更佳，在音质表现上有些许下降

只推荐网络参数`512*6`，经过大量测试这是唯一的推荐参数

`256*3`也可以正常使用

只推荐在acoustic model上使用

## WavenetAdaIN

本repo参考：https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/wavenet_adain.py

在更低的推理步数上有着更好的表现，待进一步测试，参数无需更改

训练时的显存占用有着较大的增加

导出ONNX时修改第五十行`memory_efficient=False`

目前发现在导出ONNX时会出现问题

此外`memory_efficient`在多卡情况下会出现第一个GPU显存消耗不减反增的情况，在原repo因为没有多卡场景所以不考虑这个情况

## ConvNeXt

本repo参考：https://github.com/openvpi/DiffSinger/tree/dur_diffusion

浅扩散前级使用的网络