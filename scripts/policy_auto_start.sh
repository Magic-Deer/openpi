#!/bin/bash

OPENPI_DIR=/root/Workspace/openpi
UV_CMD=/root/.local/bin/uv
API_KEY=fNR9OcNkds3zCtSXsGdhYYG7nzRZXUIGFsnCpT0eLCY=

cd $OPENPI_DIR

$UV_CMD run scripts/serve_policy.py \
  --port 6006 \
  --api_key $API_KEY \
  policy:checkpoint \
  --policy.config deerbaby_infer \
  --policy.dir $OPENPI_DIR/checkpoints/current &

sleep 20

$UV_CMD run examples/simple_client/main.py \
  --port 6006 \
  --api_key $API_KEY \
  --num_steps 2
