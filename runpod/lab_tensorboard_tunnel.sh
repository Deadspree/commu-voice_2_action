#!/usr/bin/env bash
# Run on the Mac. Leave this process running while viewing TensorBoard.
set -euo pipefail
exec ssh -N -T -J lab-jump \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:6006:127.0.0.1:6006 minhle@10.0.1.203
