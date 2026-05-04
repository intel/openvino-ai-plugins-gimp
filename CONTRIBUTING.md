# Contributing

## Dependency Management

This project follows modern Python packaging best practices:

### Runtime Dependencies
All runtime dependencies are managed in `setup.py` under the `install_requires` section. This is the single source of truth for dependencies needed to run the GIMP OpenVINO AI Plugins.

**To install the package with all runtime dependencies:**
```bash
pip install -e .
```

### Development Dependencies
Development and testing dependencies are managed in `requirements-dev.txt`. These include:
- Testing frameworks (pytest, pytest-cov, etc.)
- Code quality tools (black, flake8, isort, mypy)
- Additional testing utilities

**To install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

### Key Dependency Versions
- **Python**: Requires Python 3.10 or later (Python 3.7-3.9 are EOL)
- **OpenVINO**: Pinned to version 2026.1.0 for stability and compatibility
- **transformers**: Version range 4.37.0 to 4.56.2 for compatibility with OpenVINO
- **openvino-genai**: Pinned to 2026.1.0.0 to match OpenVINO version

### Adding New Dependencies
When adding new dependencies:
1. Add runtime dependencies to `setup.py` under `install_requires`
2. Add development/test dependencies to `requirements-dev.txt`
3. Specify version constraints when compatibility matters
4. Test installation in a clean virtual environment

## License

openvino-ai-plugins-gimp is licensed under the terms in [Apache 2.0] https://github.com/intel/openvino-ai-plugins-gimp/LICENSE.md. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.
