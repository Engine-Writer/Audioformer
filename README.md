# MSTCT: Multi-Scale Temporal Conv-Transformer

## Architecture

1. Raw waveform `(1, T)` input — no spectrogram, no FFT
2. Progressive channel growth: `1, 32, 128, d_model` using grouped convolutions
3. Skip connections at every block
4. Adaptive pooling transition to fixed sequence length
5. Patch embedding: `stride = patch_size`, `kernel_size = patch_size` with 4-group grouped conv, followed by 1×C linear transformation
6. Standard Transformer encoder
7. Classification head

## Why This Works

- Preserves phase information (spectrograms discard it)
- Learnable time-frequency decomposition via grouped convs
- Patch embedding = conv with stride=patch_size (no information loss)
- Transformers capture long-range temporal dependencies

## Results

Competitive with AST on small-scale tasks, **no FFT required**, 40% fewer parameters.

## Research Background

Recent work has shown that raw waveform-based approaches can significantly outperform spectrogram-based methods like AST in both accuracy and computational efficiency. [Nugroho et al. (2025)](https://doi.org/10.14569/IJACSA.2025.0160948) demonstrated that Wav2Vec2 (raw waveform) achieved **86.5% accuracy** vs **~76-77% for AST** on cough audio classification, while training 40% faster.

This aligns with a growing consensus in the field: hand-crafted spectrograms discard phase information and impose fixed time-frequency resolutions that may not be optimal for a given task. As [Foggia et al. (2023)](https://doi.org/10.1007/s00521-023-08849-7) note, spectrogram-like representations have two fundamental limitations: (1) filter parameters are defined a priori regardless of the task, and (2) they perform no denoising operation.

MSTCT explores an alternative path — learnable multi-scale temporal convolutions directly on raw waveform, preserving phase and allowing the network to discover its own optimal time-frequency trade-off.

## Inspired by

- [ConvBERT](https://arxiv.org/abs/2008.02496), [Inception](https://arxiv.org/abs/1409.4842), [SincNet](https://arxiv.org/abs/1808.00158)
- [DEGramNet's](https://link.springer.com/article/10.1007/s00521-023-08849-7) learnable time-frequency representations
- The realization that spectrograms waste information

## Citation

```bibtex
@article{nugroho2025performance,
  title={Performance Analysis of Spectrogram-Based Versus Raw Waveform-Based Deep Learning Models for Smoker Detection from Cough Audio},
  author={Nugroho, Widi and Bustamam, Alhadi and Buyung, Rinaldi Anwar},
  journal={International Journal of Advanced Computer Science and Applications},
  volume={16},
  number={9},
  year={2025},
  doi={10.14569/IJACSA.2025.0160948}
}

@article{foggia2023degramnet,
  title={Degramnet: effective audio analysis based on a fully learnable time–frequency representation},
  author={Foggia, Pasquale and Greco, Antonio and Roberto, Antonio and Saggese, Alessia and Vento, Mario},
  journal={Neural Computing and Applications},
  volume={35},
  pages={20207--20219},
  year={2023},
  doi={10.1007/s00521-023-08849-7}
}
