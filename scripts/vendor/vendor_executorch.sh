#!/bin/sh
# Thin wrapper: the vendoring and its --check live in vendor_executorch.py
# (Python, so the gate's --check also runs where there is no POSIX shell).
#
#   sh scripts/vendor/vendor_executorch.sh --source ET_CHECKOUT   write the tree
#   sh scripts/vendor/vendor_executorch.sh --check                verify it
#
# PYTHON picks the interpreter; it also runs upstream's codegen unless
# --codegen-python says otherwise, so it needs torchgen from torch 2.13.0.
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$here/vendor_executorch.py" "$@"
