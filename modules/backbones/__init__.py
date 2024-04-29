from modules.backbones.convnext import ConvNeXt
from modules.backbones.wavenet import WaveNet
from modules.backbones.naive_v2_diff import NaiveV2Diff
from modules.backbones.wavenet_adain import WaveNetAdaIN

BACKBONES = {
    'wavenet': WaveNet, 
    'lynxnet': NaiveV2Diff, 
    'wavenet-adain': WaveNetAdaIN, 
    'ConvNeXt': ConvNeXt
}
