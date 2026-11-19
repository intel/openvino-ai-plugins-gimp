#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Run `optimum-cli` with a Python 3.14 compatibility fix applied.

Usage: python optimum_cli_compat.py export openvino --model ... <output_dir>
(takes the same arguments as `optimum-cli`).

In Python 3.14, functools.partial objects stored as class attributes bind like
methods. optimum <= 2.3.0 stores `NormalizedConfig.with_args(...)` partials in
the NORMALIZED_CONFIG_CLASS attribute of its export configs and calls them as
`self.NORMALIZED_CONFIG_CLASS(config)`, so `self` is passed as an extra
positional argument and every export fails with:

    TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'

optimum fixed this in https://github.com/huggingface/optimum/pull/2496 (not yet
released) by wrapping the partial in staticmethod(). This script applies the
same change before optimum's export configs are defined, then runs optimum-cli.
It can be removed once a released optimum-intel works with an optimum version
that includes the fix.
"""
import functools
import sys


def _apply_python314_fix():
    if sys.version_info < (3, 14):
        return
    from optimum.utils.normalized_config import NormalizedConfig

    if isinstance(NormalizedConfig.with_args(), staticmethod):
        return  # optimum already includes the fix

    def with_args(cls, allow_new=False, **kwargs):
        return staticmethod(functools.partial(cls, allow_new=allow_new, **kwargs))

    NormalizedConfig.with_args = classmethod(with_args)


if __name__ == "__main__":
    _apply_python314_fix()
    from optimum.commands.optimum_cli import main

    sys.argv[0] = "optimum-cli"
    sys.exit(main())
