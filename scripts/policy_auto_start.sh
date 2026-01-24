#!/bin/bash

OPENPI_DIR=/root/Workspace/openpi
UV_CMD=/root/.local/bin/uv

cd $OPENPI_DIR

$UV_CMD run scripts/serve_policy.py --port 6006 --api_key fNR9OcNkds3zCtSXsGdhYYG7nzRZXUIGFsnCpT0eLCY= policy:checkpoint --policy.config caihong_v3s --policy.dir $OPENPI_DIR/checkpoints/pi0_open_close_door_caihong/f80_1216/30000 &

sleep 20

$UV_CMD run examples/simple_client/main.py --port 6006 --api_key fNR9OcNkds3zCtSXsGdhYYG7nzRZXUIGFsnCpT0eLCY= --num_steps=2

