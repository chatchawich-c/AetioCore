# Usage & Installation (`v0.1.0`)

## Python

```bash
pip install semflow semflow-dl
```

```python
import numpy as np
import semflow as sf
from semflow_dl import EdgeWhiteBoxSEM

model = sf.SEM("ind60 =~ x1 + x2 + x3\ndem60 =~ y1 + y2 + y3 + y4\ndem60 ~ ind60")
```

## R

```r
install.packages("semflow", repos = c("https://chatchawich-c.r-universe.dev", "https://cloud.r-project.org"))
library(semflow)
```
