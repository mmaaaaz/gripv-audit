# Security

## Secrets

This project reads a Hugging Face token from the environment or a local `.env`
file. `.env` is gitignored.

If you believe a token was committed:

1. Revoke it at https://huggingface.co/settings/tokens immediately.
2. Open a private security report or contact the maintainer directly.
3. Do not open a public issue containing the token.

## Dataset content

GRIP-V is a private dataset. This repository must never contain its questions,
keys, `params`, videos or frames. If you find dataset content in a commit,
report it privately so it can be removed before it is mirrored.

## Scope

This repository contains an audit harness, not a hosted service. The private
dashboard is a separate deployment with its own authentication and is outside
this repository's scope.
