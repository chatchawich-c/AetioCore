# Usage & Installation (`v0.1.2`)

## Python

```bash
pip install aetiocore-sem aetiocore-dl
```

```python
import numpy as np, pandas as pd
import aetiocore_sem as ac
from aetiocore_dl import EdgeWhiteBoxSEM, fit_whitebox_sem, intervene_do

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.normal(size=(200, 6)), columns=["x1", "x2", "x3", "y1", "y2", "y3"])
spec = ac.parse_spec("ind =~ x1 + x2 + x3\ndem =~ y1 + y2 + y3\ndem ~ ind")
fit = ac.fit_sem(spec, df, method="cb")
print(ac.fit_indices(fit))
print(ac.propagate_risk(fit, {"ind": 1.0}))
```

## R (CRAN / R-universe)

```r
install.packages("aetiocore", repos = c("https://chatchawich-c.r-universe.dev", "https://cloud.r-project.org"))
library(aetiocore)
```
