![Banner](https://github.com/AfraAmini/cpsbs/blob/main/header.jpg)

# Conditional Poisson Stochastic Beam Search

This repository contains implementation of Conditional Poisson and Sampford beam search which can be used to draw samples *without* replacement from sequence models.
*For more details, [see our paper](link).*

# Table of contents
- [Project Summary](#conditional-poisson-beams)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

# Installation
[(Back to top)](#table-of-contents)


# Usage 
[(Back to top)](#table-of-contents)

For installation instructions and basic fairseq usage instruction, please go to [the fairseq repository](https://github.com/pytorch/fairseq).
To use Stochastic Beam Search for generation:
- Add the ``--stochastic-beam-search`` option and use the (normal) ``--beam`` option with ``generate.py``
- Set the ``--nbest`` option equal to ``--beam`` (using a beam size greater than nbest is equivalent)
- For theoretical correctness of the sampling algorithm, you cannot use heuristic beam search modifications such as length normalization and early stopping. Therefore run *with* ``--no-early-stopping`` and ``--unnormalized``
- Use the ``--sampling_temperature`` option to specify the temperature used for (local) softmax normalization. For models trained using maximum likelihood, the default temperature of 1.0 does not yield good translations, so use a lower temperature.
- Example: ``python generate.py --stochastic-beam-search --beam 5 --nbest 5 --no-early-stopping --unnormalized --sampling_temperature 0.3 ...``

# Citation
[(Back to top)](#table-of-contents)

As the beam search implementation in fairseq is a bit complicated (see also [here](https://github.com/pytorch/fairseq/issues/535)), this implementation of Stochastic Beam Search has some rough edges and is not guaranteed to be compatibel with all fairseq generation parameters.
The code is not specifically optimized for memory usage etc.

If you have any comments or suggestions, please create an issue or contact me (e-mail address in the paper). If you use Stochastic Beam Search, we would be happy if you cite our paper: 
```
@inproceedings{kool2019stochastic,
  title={Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement},
  author={Wouter Kool and Herke van Hoof and Max Welling},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```