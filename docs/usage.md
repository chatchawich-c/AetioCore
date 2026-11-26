# Usage & Installation (`v0.1.1`)

## Python

```bash
pip install semflow semflow-dl
```

```python
import numpy as np, pandas as pd
import semflow as sf
from semflow_dl import EdgeWhiteBoxSEM, fit_whitebox_sem, intervene_do

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.normal(size=(200, 6)), columns=["x1", "x2", "x3", "y1", "y2", "y3"])
spec = sf.parse_spec("ind =~ x1 + x2 + x3\ndem =~ y1 + y2 + y3\ndem ~ ind")
fit = sf.fit_sem(spec, df, method="cb")
print(sf.fit_indices(fit))
print(sf.propagate_risk(fit, {"ind": 1.0}))
```

## R (CRAN / R-universe)

```r
install.packages("semflow", repos = c("https://chatchawich-c.r-universe.dev", "https://cloud.r-project.org"))
library(semflow)
```
