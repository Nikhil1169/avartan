#!/bin/bash
set -e

if ! command -v python3 >/dev/null 2>&1; then
    apt-get update
    apt-get install -y python3
fi

if ! command -v pip3 >/dev/null 2>&1; then
    apt-get update
    apt-get install -y python3-pip
fi

pip3 install --upgrade pip
pip3 install git+https://github.com/Nikhil1169/avartan.git@main

echo "INSTALL_SUCCESS"
