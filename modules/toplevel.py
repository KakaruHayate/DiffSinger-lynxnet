import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.base_module import CategorizedModule
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
    NormalInitEmbedding as Embedding
)
from modules.diffusion.ddpm import (
    GaussianDiffusion, PitchDiffusion
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.param_adaptor import ParameterAdaptorModule
from modules.fastspeech.tts_modules import RhythmRegulator, LengthRegulator
from modules.fastspeech.variance_encoder import FastSpeech2Variance
from utils.hparams import hparams


class DiffSingerAcoustic(ParameterAdaptorModule, CategorizedModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )

        self.diffusion = GaussianDiffusion(
            out_dims=out_dims,
            num_feats=1,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            denoiser_args=(
                hparams['residual_layers'],
                hparams['residual_channels']
            ),
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    def forward(
            self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
            spk_embed_id=None, gt_mel=None, infer=True, **kwargs
    ):
        condition = self.fs2(
            txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, **kwargs
        )

        if infer:
            mel_pred = self.diffusion(condition, infer=True)
            mel_pred *= ((mel2ph > 0).float()[:, :, None])
            return mel_pred
        else:
            x_recon, noise = self.diffusion(condition, gt_spec=gt_mel, infer=False)
            return x_recon, noise


class DiffSingerVariance(ParameterAdaptorModule, CategorizedModule):
    @property
    def category(self):
        return 'variance'

    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.rr = RhythmRegulator()
        self.lr = LengthRegulator()

        self.predict_pitch = hparams['predict_pitch']

        if self.predict_pitch or self.predict_variances:
            self.retake_embed = Embedding(2, hparams['hidden_size'])

        if self.predict_pitch:
            pitch_hparams = hparams['pitch_prediction_args']
            self.base_pitch_embed = Linear(1, hparams['hidden_size'])
            self.pitch_predictor = PitchDiffusion(
                vmin=pitch_hparams['pitd_norm_min'],
                vmax=pitch_hparams['pitd_norm_max'],
                cmin=pitch_hparams['pitd_clip_min'],
                cmax=pitch_hparams['pitd_clip_max'],
                repeat_bins=pitch_hparams['repeat_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args=(
                    pitch_hparams['residual_layers'],
                    pitch_hparams['residual_channels']
                )
            )

        if self.predict_variances:
            self.pitch_embed = Linear(1, hparams['hidden_size'])
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_prediction_list
            })
            self.variance_predictor = self.build_adaptor()

    def forward(
            self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, mel2ph=None,
            base_pitch=None, pitch=None, retake=None, infer=True, **kwargs
    ):
        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur, infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, None, ({} if infer else None)

        if mel2ph is None and word_dur is not None:  # inference from file
            dur_pred_align = self.rr(dur_pred_out, ph2word, word_dur)
            mel2ph = self.lr(dur_pred_align)
            mel2ph = F.pad(mel2ph, [0, base_pitch.shape[1] - mel2ph.shape[1]])

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.predict_pitch or self.predict_variances:
            if retake is None:
                retake_embed = self.retake_embed(torch.ones_like(mel2ph))
            else:
                retake_embed = self.retake_embed(retake.long())
            condition += retake_embed

        if self.predict_pitch:
            if retake is not None:
                base_pitch = base_pitch * retake + pitch * ~retake
            pitch_cond = condition + self.base_pitch_embed(base_pitch[:, :, None])
            pitch_pred_out = self.pitch_predictor(pitch_cond, pitch - base_pitch, infer)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None)

        if pitch is None:
            pitch = base_pitch + pitch_pred_out
        condition += self.pitch_embed(pitch[:, :, None])

        variance_inputs = self.collect_variance_inputs(**kwargs)
        if retake is None:
            variance_embeds = [
                self.variance_embeds[v_name](torch.zeros_like(pitch)[:, :, None])
                for v_name in self.variance_prediction_list
            ]
        else:
            variance_embeds = [
                self.variance_embeds[v_name]((v_input * ~retake)[:, :, None])
                for v_name, v_input in zip(self.variance_prediction_list, variance_inputs)
            ]
        condition += torch.stack(variance_embeds, dim=-1).sum(-1)

        variance_outputs = self.variance_predictor(condition, variance_inputs, infer)

        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out
