from modules.backbones.wavenet import WaveNet
from modules.backbones.naive_v2_diff import NaiveV2Diff

BACKBONES = {
    'wavenet': WaveNet, 
    'lynxnet': NaiveV2Diff
}
