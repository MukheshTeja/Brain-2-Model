$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
# Preload pandas before torch for the installed conda pyarrow DLL combination.
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -c "import pandas; import torch; torch.set_num_threads(1); import pytest; raise SystemExit(pytest.main(['-q']))"
exit $LASTEXITCODE
