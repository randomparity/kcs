#!/usr/bin/env bash

# Generate a random byte sequence from /dev/urandom
# Using 48 bytes because base64 encoding expands the data.
# 48 bytes of binary data results in 64 characters of base64 encoded string (48 * 4/3 = 64).
random_bytes=$(head /dev/urandom | tr -dc A-Za-z0-9_.- | head -c 48)

# Base64 encode the random bytes
# The `tr -d '\n'` removes any newline characters that base64 might add
token=$(echo -n "$random_bytes" | base64 | head -c 64)

echo "$token"
