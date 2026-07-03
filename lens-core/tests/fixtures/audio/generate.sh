#!/usr/bin/env bash
# Regenerates #41 audio test fixtures. Requires ffmpeg (+ macOS afconvert for .caf).
# Fixtures are committed so CI needs no ffmpeg; run this only to refresh them.
set -euo pipefail
cd "$(dirname "$0")"
DUR=1; FREQ=440
sine() { ffmpeg -y -v error -f lavfi -i "sine=frequency=$FREQ:duration=$DUR:sample_rate=$1" -ac "$2" "$3"; }
# Canonical source: 440Hz tone, 44.1kHz stereo (exercises resample + downmix)
sine 44100 2 tone_44100_stereo.wav
sine 48000 1 tone_48000_mono.wav
sine 16000 1 tone_16000_mono.wav          # near-identity resample
# Compressed encodings of the same tone (from the 44.1k stereo master)
SRC=tone_44100_stereo.wav
ffmpeg -y -v error -i $SRC -c:a libmp3lame -q:a 4 tone.mp3
ffmpeg -y -v error -i $SRC -c:a aac        tone_aac.m4a
ffmpeg -y -v error -i $SRC -c:a alac       tone_alac.m4a
ffmpeg -y -v error -i $SRC -c:a aac -f adts tone.aac
ffmpeg -y -v error -i $SRC -c:a flac       tone.flac
ffmpeg -y -v error -i $SRC                 tone.aiff
ffmpeg -y -v error -i $SRC -c:a libopus    tone.opus
# .caf via Apple afconvert if present, else ffmpeg
if command -v afconvert >/dev/null 2>&1; then afconvert -f caff -d LEI16 $SRC tone.caf
else ffmpeg -y -v error -i $SRC -f caf tone.caf; fi
# Degenerate fixtures for error-path tests
ffmpeg -y -v error -f lavfi -i "anullsrc=r=16000:cl=mono" -t 1 silence_16000_mono.wav  # all-zero PCM
head -c 512 /dev/urandom > corrupt.mp3     # garbage bytes with .mp3 extension
printf 'not audio at all' > unsupported.xyz

# Upsampling fixture: 8 kHz mono (ratio 2× → 16 kHz; exercises the ratio>1
# output-sizing path and the streaming iterator at a sub-target rate).
sine 8000 1 tone_8000_mono.wav

# Extended fixture: 35 s stereo (crosses the default 30 s window boundary so the
# streaming iterator emits ≥ 2 windows on the real path — used by
# streaming_multi_window_crosses_30s_boundary).
ffmpeg -y -v error -f lavfi -i "sine=frequency=$FREQ:duration=35:sample_rate=44100" \
  -ac 2 tone_44100_stereo_35s.wav

# Near-clipping fixture: amplitude ≈ 0.99 sine (sinc Gibbs ringing can push
# resampler output past ±1.0 on full-scale input; this fixture exercises the
# clamp and the output ∈ [-1,1] assertion).
ffmpeg -y -v error -f lavfi -i "sine=frequency=$FREQ:duration=$DUR:sample_rate=44100" \
  -af "volume=0.99" tone_nearclip_44100_mono.wav

# Cross-validation reference: ffmpeg swr resample of the stereo WAV to 16 kHz mono
# raw f32le. Committed so CI needs no ffmpeg. Used by cross_validates_against_ffmpeg_reference.
ffmpeg -y -v error -i tone_44100_stereo.wav -ar 16000 -ac 1 -f f32le tone_44100_stereo.ref16k.f32le

# Anti-aliasing fixture: 12 kHz tone at 44.1 kHz. A broken low-pass would let this alias
# to |16000-12000|=4000 Hz in the output. Used by sinc_filter_rejects_above_nyquist_alias.
ffmpeg -y -v error -f lavfi -i "sine=frequency=12000:duration=1:sample_rate=44100" -ac 1 tone_12k_44100_mono.wav

echo "fixtures regenerated in $(pwd)"
