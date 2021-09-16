![Banner](https://github.com/AfraAmini/cpsbs/blob/main/header.jpg)

# Conditional Poisson Stochastic Beam Search
![GitHub last commit](https://img.shields.io/github/last-commit/AfraAmini/cpsbs)
![GitHub](https://img.shields.io/github/license/AfraAmini/cpsbs)

This repository contains implementation of Conditional Poisson Stochastic Beam Search, which can be used to draw samples *without* replacement from sequence models.
*For more details, [see our paper](link).*

# Table of contents
- [Project Summary](#conditional-poisson-beams)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

# Installation
[(Back to top)](#table-of-contents)

For detailed installation instructions and basic fairseq usage instruction, please go to [the fairseq repository](https://github.com/pytorch/fairseq).

Basically, to install and develop fairseq locally:
```bash
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
*Note*: If you are using MacOS `Xpreprocessor` argument should be added to 
the setup of CPS dynamic program. So the extension definition in `setup.py` file
should look like this:
```python
cps = [Extension('fairseq.cps_dp', ["fairseq/cps_dp.pyx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                 extra_link_args=['-Xpreprocessor', '-fopenmp']
                 )]
```

# Usage 
[(Back to top)](#table-of-contents)

Example:
```bash
python generate.py --cps --num-experiments 3 
                    --beam 5 --nbest 5 
                    --nucleus-threshold 0.99
                    --unnormalized --sampling-temperature 0.1 
                    [DATAPATH] --path [MODELPATH]
```
- ``--cps``: to use CPSBS for decoding
- ``-num-experiments``: repeat the procedure for the specified number of times. Useful for building estimators.
- ``--beam``: beam size or sample size
- ``--nbest``: equal to ``--beam`` (using a beam size greater than nbest is equivalent)
- ``--nucleus-threshold``: probability threshold for nucleus filtering. Default is 1. (no filtering)
- ``--no-early-stopping`` and ``--unnormalized``: for theoretical correctness of the sampling algorithm 
- ``--sampling_temperature``: temperature used for local softmax normalization. 

# Citation
[(Back to top)](#table-of-contents)

If you have any comments or suggestions, please let us know by creating an issue or contacting me. If you use CPSBS, we would be happy if you cite our paper: 
```

```
