#!/usr/bin/env sh
# Lightweight healthcheck — used by Docker HEALTHCHECK and external monitors.
curl -sf http://localhost:7860/ > /dev/null
