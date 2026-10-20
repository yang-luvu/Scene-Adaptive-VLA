#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/CARLA_0.9.15" >&2
    exit 2
fi

CARLA_PATH="$1"
mkdir -p "$CARLA_PATH"
cd "$CARLA_PATH"

# Download the 15–20 GB simulator archive.
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

tar -xvf CARLA_0.9.15.tar.gz

rm CARLA_0.9.15.tar.gz

# Import the additional maps.
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. 
bash ImportAssets.sh
