//! SNAC 24 kHz neural-codec decoder (issue #191 [161c]).
//!
//! Ports the *decode* path of `hubertsiuzdak/snac_24khz` to candle: an Orpheus
//! audio-token stream → three hierarchical codebook frames → 24 kHz mono PCM.
//! The encoder/quantizer-search paths are not needed (we only decode).
//!
//! Weight-norm is folded at load so no runtime norm param is kept (see
//! [`fold_weight_norm`]). Decode runs on CPU for cross-platform determinism.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module};

use crate::error::LensError;
use crate::tts::AudioBuffer;
use crate::tts::audio::TARGET_RATE;

pub const SNAC_MODEL_ID: &str = "snac";
pub const SNAC_MODEL_URL: &str =
    "https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/pytorch_model.bin";
pub const SNAC_MODEL_SHA256_HEX: &str =
    "4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40";
pub const SNAC_MODEL_RELPATH: &str = "models/snac/pytorch_model.bin";

// Orpheus audio-token framing (raw token-id math, from the spike reference decoder):
// `code = raw_id - 128266 - ((kept_pos % 7) * 4096)`, kept iff `0 < code < 4096`.
// The leading markers (start-of-AI 128261, start-of-audio 128257) and the
// end-of-audio marker (128258) all fall out of range and are skipped naturally.
const CUSTOM_TOKEN_BASE: i64 = 128266;
const CODEBOOK_SPAN: i64 = 4096;
const TOKENS_PER_FRAME: usize = 7;

type Weights = HashMap<String, Tensor>;
type CResult<T> = candle_core::Result<T>;

fn missing(name: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("snac: missing tensor `{name}`"))
}

fn get_t(w: &Weights, name: &str) -> CResult<Tensor> {
    w.get(name).cloned().ok_or_else(|| missing(name))
}

/// Folds weight-norm at load: `w = v · (g / ‖v‖)`, where `‖v‖` is the L2 norm
/// over every dim except the output dim (0) — matching PyTorch `weight_norm(dim=0)`.
/// `g` = checkpoint `original0` (shape `[C,1,1]`), `v` = `original1` (full weight).
fn fold_weight_norm(g: &Tensor, v: &Tensor) -> CResult<Tensor> {
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let scale = g.broadcast_div(&norm)?;
    v.broadcast_mul(&scale)
}

fn wn_conv1d(w: &Weights, prefix: &str, cfg: Conv1dConfig) -> CResult<Conv1d> {
    let g = get_t(w, &format!("{prefix}.parametrizations.weight.original0"))?;
    let v = get_t(w, &format!("{prefix}.parametrizations.weight.original1"))?;
    let weight = fold_weight_norm(&g, &v)?;
    let bias = get_t(w, &format!("{prefix}.bias"))?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

fn wn_conv_transpose1d(
    w: &Weights,
    prefix: &str,
    cfg: ConvTranspose1dConfig,
) -> CResult<ConvTranspose1d> {
    let g = get_t(w, &format!("{prefix}.parametrizations.weight.original0"))?;
    let v = get_t(w, &format!("{prefix}.parametrizations.weight.original1"))?;
    let weight = fold_weight_norm(&g, &v)?;
    let bias = get_t(w, &format!("{prefix}.bias"))?;
    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

fn conv1d_cfg(padding: usize, dilation: usize, groups: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding,
        stride: 1,
        dilation,
        groups,
        ..Default::default()
    }
}

/// `Snake1d`: `x + sin²(αx) / (α + 1e-9)`, per-channel learnable α (shape `[1,C,1]`).
/// The `+1e-9` guards the reciprocal as α→0 (matches the reference `snake`).
struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn load(w: &Weights, name: &str) -> CResult<Self> {
        Ok(Self {
            alpha: get_t(w, name)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?.sqr()?;
        let inv = self.alpha.affine(1.0, 1e-9)?.recip()?;
        x.broadcast_add(&s.broadcast_mul(&inv)?)
    }
}

/// `ResidualUnit`: Snake → dilated depthwise Conv1d(k7) → Snake → Conv1d(k1),
/// added back to the (center-cropped) input. Convs are length-preserving here
/// (`pad = 3·dilation`), so the crop is a no-op but kept for fidelity.
struct ResidualUnit {
    snake1: Snake,
    conv1: Conv1d,
    snake2: Snake,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn load(w: &Weights, prefix: &str, dim: usize, dilation: usize) -> CResult<Self> {
        let pad = 3 * dilation;
        Ok(Self {
            snake1: Snake::load(w, &format!("{prefix}.block.0.alpha"))?,
            conv1: wn_conv1d(
                w,
                &format!("{prefix}.block.1"),
                conv1d_cfg(pad, dilation, dim),
            )?,
            snake2: Snake::load(w, &format!("{prefix}.block.2.alpha"))?,
            conv2: wn_conv1d(w, &format!("{prefix}.block.3"), conv1d_cfg(0, 1, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let y = self.snake1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.snake2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        let (xt, yt) = (x.dim(2)?, y.dim(2)?);
        let x = if xt > yt {
            x.narrow(2, (xt - yt) / 2, yt)?
        } else {
            x.clone()
        };
        x.add(&y)
    }
}

/// One decoder upsampling stage: Snake → ConvTranspose1d(×stride) → [NoiseBlock]
/// → 3 ResidualUnits (dilations 1/3/9).
///
/// NoiseBlock is intentionally omitted: it adds `randn·linear(x)` (≈ −42 dB
/// texture), and zeroing the noise (per the plan's determinism contract) makes
/// `x + 0·… = x`, i.e. a no-op. End-to-end variety is preserved by stochastic
/// Orpheus generation, and dropping it keeps the decoder bit-reproducible.
struct DecoderBlock {
    snake: Snake,
    up: ConvTranspose1d,
    residuals: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(w: &Weights, prefix: &str, output_dim: usize, stride: usize) -> CResult<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: stride.div_ceil(2),
            output_padding: stride % 2,
            stride,
            dilation: 1,
            groups: 1,
        };
        let residuals = [(3usize, 1usize), (4, 3), (5, 9)]
            .into_iter()
            .map(|(idx, dilation)| {
                ResidualUnit::load(w, &format!("{prefix}.block.{idx}"), output_dim, dilation)
            })
            .collect::<CResult<Vec<_>>>()?;
        Ok(Self {
            snake: Snake::load(w, &format!("{prefix}.block.0.alpha"))?,
            up: wn_conv_transpose1d(w, &format!("{prefix}.block.1"), cfg)?,
            residuals,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut x = self.snake.forward(x)?;
        x = self.up.forward(&x)?;
        for r in &self.residuals {
            x = r.forward(&x)?;
        }
        Ok(x)
    }
}

struct Decoder {
    conv_in_dw: Conv1d,
    conv_in_pw: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: Conv1d,
}

impl Decoder {
    fn load(w: &Weights, cfg: &SnacConfig) -> CResult<Self> {
        let latent = cfg.latent_dim;
        let conv_in_dw = wn_conv1d(w, "decoder.model.0", conv1d_cfg(3, 1, latent))?;
        let conv_in_pw = wn_conv1d(w, "decoder.model.1", conv1d_cfg(0, 1, 1))?;

        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            let output_dim = cfg.decoder_dim >> (i + 1);
            blocks.push(DecoderBlock::load(
                w,
                &format!("decoder.model.{}", i + 2),
                output_dim,
                stride,
            )?);
        }

        let snake_idx = cfg.decoder_rates.len() + 2;
        let snake_out = Snake::load(w, &format!("decoder.model.{snake_idx}.alpha"))?;
        let conv_out = wn_conv1d(
            w,
            &format!("decoder.model.{}", snake_idx + 1),
            conv1d_cfg(3, 1, 1),
        )?;

        Ok(Self {
            conv_in_dw,
            conv_in_pw,
            blocks,
            snake_out,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let mut x = self.conv_in_dw.forward(x)?;
        x = self.conv_in_pw.forward(&x)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        x = self.snake_out.forward(&x)?;
        x = self.conv_out.forward(&x)?;
        x.tanh()
    }
}

/// One RVQ level used at decode time: an embedding codebook plus the `out_proj`
/// (kernel-1 conv) that lifts the 8-dim code back to the latent dim, then a
/// stride-repeat upsample. `in_proj` / the codebook search are decode-unused.
struct QuantizerLevel {
    codebook: Tensor,
    out_proj: Conv1d,
    stride: usize,
}

impl QuantizerLevel {
    fn load(w: &Weights, i: usize, stride: usize) -> CResult<Self> {
        let prefix = format!("quantizer.quantizers.{i}");
        Ok(Self {
            codebook: get_t(w, &format!("{prefix}.codebook.weight"))?,
            out_proj: wn_conv1d(w, &format!("{prefix}.out_proj"), conv1d_cfg(0, 1, 1))?,
            stride,
        })
    }

    fn decode(&self, codes: &[u32], codebook_size: usize, device: &Device) -> CResult<Tensor> {
        if let Some(&bad) = codes.iter().find(|&&c| c as usize >= codebook_size) {
            return Err(candle_core::Error::Msg(format!(
                "snac: code {bad} out of range for codebook size {codebook_size}"
            )));
        }
        let idx = Tensor::from_vec(codes.to_vec(), (codes.len(),), device)?;
        // [T, codebook_dim] -> [codebook_dim, T] -> [1, codebook_dim, T]
        let emb = self.codebook.index_select(&idx, 0)?.t()?.unsqueeze(0)?;
        let projected = self.out_proj.forward(&emb)?;
        repeat_interleave_last(&projected, self.stride)
    }
}

/// Repeats each time-step `n` times along the last dim (candle has no
/// `repeat_interleave`): `[B,C,T] -> [B,C,T·n]`.
fn repeat_interleave_last(x: &Tensor, n: usize) -> CResult<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, c, t) = x.dims3()?;
    x.unsqueeze(3)?
        .broadcast_as((b, c, t, n))?
        .contiguous()?
        .reshape((b, c, t * n))
}

#[derive(Debug, Clone)]
pub struct SnacConfig {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub vq_strides: Vec<usize>,
}

impl SnacConfig {
    pub fn snac_24khz() -> Self {
        Self {
            codebook_size: 4096,
            codebook_dim: 8,
            latent_dim: 768,
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            vq_strides: vec![4, 2, 1],
        }
    }
}

pub struct SnacDecoder {
    quantizers: Vec<QuantizerLevel>,
    decoder: Decoder,
    config: SnacConfig,
    device: Device,
}

impl SnacDecoder {
    /// Loads the real `snac_24khz` decoder from the upstream PyTorch `.bin`.
    pub fn load(path: &Path) -> Result<Self, LensError> {
        // Generic message: the underlying candle error embeds the on-disk path,
        // which must not cross the IPC boundary.
        let tensors = candle_core::pickle::read_all(path)
            .map_err(|_| LensError::Tts("snac: reading weights failed".into()))?;
        let weights: Weights = tensors.into_iter().collect();
        Self::from_weights(weights, SnacConfig::snac_24khz(), Device::Cpu)
            .map_err(|e| LensError::Tts(format!("snac: building decoder failed: {e}")))
    }

    fn from_weights(weights: Weights, config: SnacConfig, device: Device) -> CResult<Self> {
        let weights = to_f32(weights, &device)?;
        let quantizers = config
            .vq_strides
            .iter()
            .enumerate()
            .map(|(i, &stride)| QuantizerLevel::load(&weights, i, stride))
            .collect::<CResult<Vec<_>>>()?;
        let decoder = Decoder::load(&weights, &config)?;
        Ok(Self {
            quantizers,
            decoder,
            config,
            device,
        })
    }

    /// Decodes an Orpheus audio-token stream to 24 kHz mono PCM. Full-batch (no
    /// streaming window). Returns an error if fewer than one full 7-token frame
    /// survives extraction.
    pub fn decode(&self, tokens: &[u32]) -> Result<AudioBuffer, LensError> {
        let samples = self
            .decode_inner(tokens)
            .map_err(|e| LensError::Tts(format!("snac: decode failed: {e}")))?;
        Ok(AudioBuffer::mono(samples, TARGET_RATE))
    }

    fn decode_inner(&self, tokens: &[u32]) -> CResult<Vec<f32>> {
        let codes = extract_codes(tokens);
        let frames = regroup_frames(&codes).ok_or_else(|| {
            candle_core::Error::Msg("snac: fewer than one full 7-token frame".to_string())
        })?;
        let z_q = self.compose_latent(&frames)?;
        let audio = self.decoder.forward(&z_q)?;
        audio.flatten_all()?.to_vec1::<f32>()
    }

    /// Assembles the decoder-input latent from the 3 codebook frames (the SNAC
    /// `from_codes` path): per level, embed → `out_proj` → stride-repeat, summed.
    fn compose_latent(&self, frames: &[Vec<u32>; 3]) -> CResult<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (level, codes) in self.quantizers.iter().zip(frames.iter()) {
            let up = level.decode(codes, self.config.codebook_size, &self.device)?;
            acc = Some(match acc {
                None => up,
                Some(prev) => prev.add(&up)?,
            });
        }
        acc.ok_or_else(|| candle_core::Error::Msg("snac: no quantizer levels".to_string()))
    }
}

fn to_f32(weights: Weights, device: &Device) -> CResult<Weights> {
    weights
        .into_iter()
        .map(|(k, t)| {
            let t = t.to_dtype(DType::F32)?.to_device(device)?;
            Ok((k, t))
        })
        .collect()
}

/// Extracts SNAC codes from a raw Orpheus token-id stream. Stateful over *kept*
/// tokens: position advances only when a token survives the range check, so
/// markers are skipped without consuming a frame slot (matches the reference).
pub(crate) fn extract_codes(raw_ids: &[u32]) -> Vec<u32> {
    let mut codes = Vec::new();
    let mut pos = 0usize;
    for &raw in raw_ids {
        let code =
            raw as i64 - CUSTOM_TOKEN_BASE - ((pos % TOKENS_PER_FRAME) as i64) * CODEBOOK_SPAN;
        // Codebook indices are 0..CODEBOOK_SPAN-1; the upper bound is EXCLUSIVE so a
        // stray `code == CODEBOOK_SPAN` is dropped gracefully here rather than aborting
        // the whole overview at the codebook lookup. `code > 0` matches the reference
        // `token > 0` filter, which intentionally drops code 0.
        if code > 0 && code < CODEBOOK_SPAN {
            codes.push(code as u32);
            pos += 1;
        }
    }
    codes
}

/// Regroups a flat code stream into the 3 hierarchical codebook frames
/// (`vq_strides = [4,2,1]`, 7 codes/frame): `c0=[f0]`, `c1=[f1,f4]`,
/// `c2=[f2,f3,f5,f6]`. Returns `None` if fewer than one full frame is present.
pub(crate) fn regroup_frames(codes: &[u32]) -> Option<[Vec<u32>; 3]> {
    let num_frames = codes.len() / TOKENS_PER_FRAME;
    if num_frames == 0 {
        return None;
    }
    let mut c0 = Vec::with_capacity(num_frames);
    let mut c1 = Vec::with_capacity(2 * num_frames);
    let mut c2 = Vec::with_capacity(4 * num_frames);
    for j in 0..num_frames {
        let i = TOKENS_PER_FRAME * j;
        c0.push(codes[i]);
        c1.push(codes[i + 1]);
        c1.push(codes[i + 4]);
        c2.push(codes[i + 2]);
        c2.push(codes[i + 3]);
        c2.push(codes[i + 5]);
        c2.push(codes[i + 6]);
    }
    Some([c0, c1, c2])
}

// Micro-golden vector for `tiny_decode_micro_golden` — the deterministic 1-frame
// output of the tiny synthetic decoder. Regenerate ONLY on an intentional change
// to the tiny config/weights (run the test; it prints the actual samples on fail).
#[cfg(test)]
const SNAC_MICRO_GOLDEN: [f32; 16] = [
    0.314442, 0.28464544, 0.25355858, 0.20777695, 0.20949014, 0.2002126, 0.19394012, 0.19240302,
    0.19853172, 0.21379876, 0.22166005, 0.22113404, 0.2216395, 0.2123655, 0.20707607, 0.21904239,
];

#[cfg(test)]
mod tests {
    use super::*;

    // Builds a raw Orpheus token id that decodes to `code` at kept-position `pos`.
    fn raw_id(code: i64, pos: usize) -> u32 {
        (code + CUSTOM_TOKEN_BASE + (pos % TOKENS_PER_FRAME) as i64 * CODEBOOK_SPAN) as u32
    }

    #[test]
    fn extract_codes_skips_leading_and_trailing_markers() {
        // start-of-AI, start-of-audio, one 7-token frame [1..=7], end-of-audio.
        let mut raw = vec![128261u32, 128257];
        for (p, code) in (1..=7).enumerate() {
            raw.push(raw_id(code as i64, p));
        }
        raw.push(128258);
        assert_eq!(extract_codes(&raw), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn extract_codes_keep_rule_boundaries() {
        // Keep rule is `0 < code < 4096`: code 0 (raw = base) is dropped; code 4095
        // (max valid index) is kept; code == 4096 and 4097 are dropped (out of range).
        assert_eq!(extract_codes(&[raw_id(0, 0)]), Vec::<u32>::new());
        assert_eq!(
            extract_codes(&[CUSTOM_TOKEN_BASE as u32 + 4095]),
            vec![4095]
        );
        assert_eq!(
            extract_codes(&[CUSTOM_TOKEN_BASE as u32 + 4096]),
            Vec::<u32>::new()
        );
        assert_eq!(
            extract_codes(&[CUSTOM_TOKEN_BASE as u32 + 4097]),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn extract_codes_position_only_advances_on_kept() {
        // A dropped token (marker) between two audio tokens must NOT shift the
        // position offset applied to the following kept token.
        let raw = vec![raw_id(1, 0), 128258, raw_id(2, 1)];
        assert_eq!(extract_codes(&raw), vec![1, 2]);
    }

    #[test]
    fn regroup_frames_single_frame_reorg() {
        let frames = regroup_frames(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(frames[0], vec![1]);
        assert_eq!(frames[1], vec![2, 5]);
        assert_eq!(frames[2], vec![3, 4, 6, 7]);
    }

    #[test]
    fn regroup_frames_two_frames_and_truncation() {
        // 15 codes -> 2 full frames (last stray code dropped).
        let codes: Vec<u32> = (1..=15).collect();
        let frames = regroup_frames(&codes).unwrap();
        assert_eq!(frames[0], vec![1, 8]);
        assert_eq!(frames[1], vec![2, 5, 9, 12]);
        assert_eq!(frames[2], vec![3, 4, 6, 7, 10, 11, 13, 14]);
    }

    #[test]
    fn regroup_frames_needs_full_frame() {
        assert!(regroup_frames(&[1, 2, 3, 4, 5, 6]).is_none());
        assert!(regroup_frames(&[]).is_none());
    }

    #[test]
    fn fold_weight_norm_matches_hand_computation() {
        let device = Device::Cpu;
        // v = [[[3, 4]]] (‖v‖ = 5), g = [[[10]]]  ->  w = v * 10/5 = [6, 8].
        let v = Tensor::from_vec(vec![3f32, 4.0], (1, 1, 2), &device).unwrap();
        let g = Tensor::from_vec(vec![10f32], (1, 1, 1), &device).unwrap();
        let w = fold_weight_norm(&g, &v).unwrap();
        let got = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((got[0] - 6.0).abs() < 1e-5, "got {got:?}");
        assert!((got[1] - 8.0).abs() < 1e-5, "got {got:?}");
    }

    #[test]
    fn conv_transpose_upsamples_exactly_by_stride() {
        // Assert the ConvTranspose1d stride/padding/output_padding convention
        // (kernel=2·stride, padding=ceil(stride/2), output_padding=stride%2)
        // yields an exact ×stride length change — the property that guards
        // against mis-padded upsampling emitting garbage that still "type-checks".
        let device = Device::Cpu;
        for stride in [2usize, 3, 4, 8] {
            let k = 2 * stride;
            let cfg = ConvTranspose1dConfig {
                padding: stride.div_ceil(2),
                output_padding: stride % 2,
                stride,
                dilation: 1,
                groups: 1,
            };
            let weight = Tensor::ones((1, 1, k), DType::F32, &device).unwrap();
            let bias = Tensor::zeros((1,), DType::F32, &device).unwrap();
            let ct = ConvTranspose1d::new(weight, Some(bias), cfg);
            let l_in = 5usize;
            let x = Tensor::ones((1, 1, l_in), DType::F32, &device).unwrap();
            let out = ct.forward(&x).unwrap();
            assert_eq!(out.dim(2).unwrap(), stride * l_in, "stride {stride}");
        }
    }

    // --- Tiny synthetic decoder: exercises the full conv stack offline (no
    // weights/network), with deterministic hand-set weights. ---

    fn tiny_config() -> SnacConfig {
        SnacConfig {
            codebook_size: 8,
            codebook_dim: 2,
            latent_dim: 3,
            decoder_dim: 4,
            decoder_rates: vec![2, 2],
            vq_strides: vec![4, 2, 1],
        }
    }

    // Deterministic weight for a tensor of the given shape: a small, bounded,
    // index-dependent ramp so every element is distinct and reproducible.
    fn synth(shape: &[usize], seed: f32, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| 0.6 * ((i as f32 * 0.37 + seed).sin()))
            .collect();
        Tensor::from_vec(data, shape.to_vec(), device).unwrap()
    }

    fn bump(s: &mut f32) -> f32 {
        *s += 1.0;
        *s
    }

    // g shape [out,1,1]; v shape [out, in, k]; bias [out].
    fn put_wn(
        w: &mut Weights,
        prefix: &str,
        out: usize,
        inn: usize,
        k: usize,
        s: &mut f32,
        device: &Device,
    ) {
        w.insert(
            format!("{prefix}.parametrizations.weight.original0"),
            synth(&[out, 1, 1], bump(s), device),
        );
        w.insert(
            format!("{prefix}.parametrizations.weight.original1"),
            synth(&[out, inn, k], bump(s), device),
        );
        w.insert(format!("{prefix}.bias"), synth(&[out], bump(s), device));
    }

    // Populates a weight map for `tiny_config` with the parametrization keys the
    // loader expects (original0 = g, original1 = v, bias, alpha, codebook).
    fn tiny_weights(device: &Device) -> Weights {
        let cfg = tiny_config();
        let mut w = Weights::new();
        let s = &mut 0.0f32;

        for i in 0..cfg.vq_strides.len() {
            let p = format!("quantizer.quantizers.{i}");
            w.insert(
                format!("{p}.codebook.weight"),
                synth(&[cfg.codebook_size, cfg.codebook_dim], bump(s), device),
            );
            put_wn(
                &mut w,
                &format!("{p}.out_proj"),
                cfg.latent_dim,
                cfg.codebook_dim,
                1,
                s,
                device,
            );
        }

        put_wn(&mut w, "decoder.model.0", cfg.latent_dim, 1, 7, s, device);
        put_wn(
            &mut w,
            "decoder.model.1",
            cfg.decoder_dim,
            cfg.latent_dim,
            1,
            s,
            device,
        );

        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            let input_dim = cfg.decoder_dim >> i;
            let output_dim = cfg.decoder_dim >> (i + 1);
            let p = format!("decoder.model.{}", i + 2);
            w.insert(
                format!("{p}.block.0.alpha"),
                synth(&[1, input_dim, 1], bump(s), device),
            );
            // ConvTranspose weight: [in, out, k].
            w.insert(
                format!("{p}.block.1.parametrizations.weight.original0"),
                synth(&[input_dim, 1, 1], bump(s), device),
            );
            w.insert(
                format!("{p}.block.1.parametrizations.weight.original1"),
                synth(&[input_dim, output_dim, 2 * stride], bump(s), device),
            );
            w.insert(
                format!("{p}.block.1.bias"),
                synth(&[output_dim], bump(s), device),
            );
            for idx in [3usize, 4, 5] {
                let rp = format!("{p}.block.{idx}");
                w.insert(
                    format!("{rp}.block.0.alpha"),
                    synth(&[1, output_dim, 1], bump(s), device),
                );
                put_wn(
                    &mut w,
                    &format!("{rp}.block.1"),
                    output_dim,
                    1,
                    7,
                    s,
                    device,
                );
                w.insert(
                    format!("{rp}.block.2.alpha"),
                    synth(&[1, output_dim, 1], bump(s), device),
                );
                put_wn(
                    &mut w,
                    &format!("{rp}.block.3"),
                    output_dim,
                    output_dim,
                    1,
                    s,
                    device,
                );
            }
        }

        let final_dim = cfg.decoder_dim >> cfg.decoder_rates.len();
        let snake_idx = cfg.decoder_rates.len() + 2;
        w.insert(
            format!("decoder.model.{snake_idx}.alpha"),
            synth(&[1, final_dim, 1], bump(s), device),
        );
        put_wn(
            &mut w,
            &format!("decoder.model.{}", snake_idx + 1),
            1,
            final_dim,
            7,
            s,
            device,
        );
        w
    }

    fn tiny_tokens(num_frames: usize) -> Vec<u32> {
        let mut raw = Vec::new();
        for f in 0..num_frames {
            for slot in 0..7 {
                let pos = f * 7 + slot;
                // codes cycle within [1, codebook_size) so index_select stays valid.
                let code = 1 + (pos % 6) as i64;
                raw.push(raw_id(code, pos));
            }
        }
        raw
    }

    #[test]
    fn tiny_decode_output_length_matches_upsampling_product() {
        let device = Device::Cpu;
        let dec = SnacDecoder::from_weights(tiny_weights(&device), tiny_config(), device).unwrap();
        for num_frames in [1usize, 2] {
            let audio = dec.decode(&tiny_tokens(num_frames)).unwrap();
            // base T = vq_strides[0] * num_frames; decoder upsamples by ∏rates.
            let expected = tiny_config().vq_strides[0]
                * num_frames
                * tiny_config().decoder_rates.iter().product::<usize>();
            assert_eq!(audio.samples.len(), expected, "frames {num_frames}");
            assert_eq!(audio.sample_rate, TARGET_RATE);
            assert_eq!(audio.channels, 1);
            assert!(audio.samples.iter().all(|s| s.is_finite()));
            // Tanh output is strictly bounded.
            assert!(audio.samples.iter().all(|s| s.abs() <= 1.0));
        }
    }

    // Micro-golden (AC1.2): the tiny synthetic decoder is fully deterministic, so
    // its 1-frame output is a stable numeric fingerprint. A per-sample max-abs
    // drift ≥ 1e-3 means a conv/padding/activation regression. Golden values are
    // this decoder's committed output (regenerated only on an intentional change).
    #[test]
    fn tiny_decode_micro_golden() {
        let device = Device::Cpu;
        let dec = SnacDecoder::from_weights(tiny_weights(&device), tiny_config(), device).unwrap();
        let audio = dec.decode(&tiny_tokens(1)).unwrap();
        let golden: [f32; 16] = SNAC_MICRO_GOLDEN;
        assert_eq!(audio.samples.len(), golden.len());
        let max_abs = audio
            .samples
            .iter()
            .zip(golden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "max-abs error {max_abs}; got {:?}",
            audio.samples
        );
    }

    fn read_le_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_le_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // Bit-exact parity golden (AC1.3): loads the REAL snac_24khz weights and decodes
    // the committed fixed Orpheus token stream, asserting per-sample agreement with a
    // committed golden produced by the upstream Python `snac` decoder run with every
    // NoiseBlock zeroed (identity forward) — the same determinism contract the port
    // enforces. This exercises the whole chain (extract → regroup → compose → decode).
    // Gated on `LENS_RUN_MODEL_TESTS=1` and `LENS_SNAC_WEIGHTS` (the upstream
    // `pytorch_model.bin`, never committed).
    //
    // TOL is evidence-based: the measured max-abs diff vs the golden is ~2.2e-7
    // (printed below) — i.e. bit-exact to f32 rounding. TOL keeps ~20× margin for
    // cross-platform BLAS accumulation-order variance; a real port regression
    // (padding, upsample, weight-norm dim, noise) would exceed it by many orders.
    #[test]
    fn real_snac_parity_golden() {
        const TOL: f32 = 5.0e-6;
        if std::env::var("LENS_RUN_MODEL_TESTS").is_err() {
            eprintln!("skipping real_snac_parity_golden (set LENS_RUN_MODEL_TESTS=1)");
            return;
        }
        let Ok(path) = std::env::var("LENS_SNAC_WEIGHTS") else {
            eprintln!("skipping: set LENS_SNAC_WEIGHTS to the snac pytorch_model.bin path");
            return;
        };
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
        let tokens = read_le_u32(
            &std::fs::read(format!("{dir}snac_parity_tokens.u32")).expect("read tokens fixture"),
        );
        let golden = read_le_f32(
            &std::fs::read(format!("{dir}snac_parity_golden.f32")).expect("read golden fixture"),
        );

        let dec = SnacDecoder::load(Path::new(&path)).expect("load real snac weights");
        let audio = dec.decode(&tokens).expect("decode");
        assert_eq!(
            audio.samples.len(),
            golden.len(),
            "sample-count mismatch vs golden"
        );
        assert!(
            audio
                .samples
                .iter()
                .all(|s| s.is_finite() && s.abs() <= 1.0)
        );

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (a, g) in audio.samples.iter().zip(golden.iter()) {
            let d = (a - g).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean_abs = sum_abs / golden.len() as f64;
        let rms =
            (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
        eprintln!(
            "snac parity: samples={} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rms={rms:.6}",
            audio.samples.len()
        );
        assert!(
            rms > 1e-4,
            "decoded audio is effectively silent (rms {rms})"
        );
        assert!(
            max_abs < TOL,
            "max-abs error {max_abs:.6e} exceeds TOL {TOL:.6e} — likely a port regression \
             (padding/output_padding, upsample, weight-norm dim, or NoiseBlock not zeroed)"
        );
    }
}
