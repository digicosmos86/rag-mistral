!#/usr/bin/env bash

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport="8230"
ipnip="$(hostname -i)"
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L ${ipnport}:${ipnip}:${ipnport} ${USER}@ssh.ccv.brown.edu
    -----------------------------------------------------------------
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    http://localhost:${ipnport}  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
module load python/3.11.0 openssl/3.0.0
module load gcc/10.2 cuda/11.3.1 cudnn/8.2.0

poetry run streamlit run --server.port ${ipnport} --server.address ${ipnip} app/ragbot.py
