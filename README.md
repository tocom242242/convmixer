# ConvMixer

This is an unofficial implementation of ConvMixer.
(It closely follows the source code from the paper.)
Paper: https://arxiv.org/pdf/2201.09792.pdf


Note: This implementation has only been tested on CPU so far.

## Prerequisites

- Python 3.10 or higher
- poetry 1.3.1

## Installation

First, clone this repository:

```
git clone https://github.com/tocom242242/convmixer.git
cd convmixer
```

Next, use `poetry` to install the dependencies:

```
poetry install
```

## How To Use

``` pytorch
import torch
from convmixer.convmixer import ConvMixer

model = ConvMixer(dim=3, depth=7)
x = torch.rand(1, 3, 32, 32)
model(x)
```
