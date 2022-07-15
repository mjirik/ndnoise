[![Build Status](https://travis-ci.org/mjirik/ndnoise.svg?branch=master)](https://travis-ci.org/mjirik/ndnoise)
[![Coverage Status](https://coveralls.io/repos/github/mjirik/ndnoise/badge.svg)](https://coveralls.io/github/mjirik/ndnoise)

# ndnoise
Noise generator

# Example

```python
import ndnoise
from matplotlib import pyplot as plt

noise = ndnoise.generator.noises(
        [201, 202, 203],
        sample_spacing=[1,1,1],
        random_generator_seed=5,
        lambda0=1,
        lambda1=16,
        exponent=0,
        method="space"
    )
plt.imshow(noise[:,:,100])
```


# References

[Noise function and map generation](http://www.redblobgames.com/articles/noise/introduction.html)

perlin - clasic noise
simplex - better noise

opensimplex - open 