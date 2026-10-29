# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Pin past the eMudhra/emSign incident (discuss.python.org/t/109576). A floating
# :latest tag can stay cached on a pre-0.13.4 digest.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app

# Distrust emSign ECC Root CA - G3 and the Comodo AAA cross-sign root, then
# replace the base image's pip. pip < 26.2.2 trusts its vendored bundle in
# addition to the system store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i \
        -e 's|^mozilla/emSign_ECC_Root_CA_-_G3|!mozilla/emSign_ECC_Root_CA_-_G3|' \
        -e 's|^mozilla/Comodo_AAA|!mozilla/Comodo_AAA|' \
        /etc/ca-certificates.conf \
    && update-ca-certificates --fresh >/dev/null \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image).
COPY . .

RUN uv pip install --system --no-cache "certifi>=2026.10.19" ".[server]"

EXPOSE 8000
CMD ["uvicorn", "argus_darkroom.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

