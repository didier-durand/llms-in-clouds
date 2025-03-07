#!/bin/bash
printf "\nstarting for SGLang...\n"

printf "\n### displaying SGLang  environment variables ...\n"
echo "Python version: ${PYTHON_VERSION:=3.12}"
echo "Log level: ${SGL_LOG_LEVEL:=info}"
echo "Host: ${SGL_HOST:=0.0.0.0}"
echo "Port: ${SGL_PORT:=30000}"
echo "Model dir: ${SGL_MODEL_DIR}"
echo "Model: ${SGL_MODEL}"
echo "SGLang tp size: ${SGL_TP_SIZE}"
echo "SGLang params: ${SGL_PARAMS:=--enable-metrics --trust-remote-code --enable-p2p-check --disable-cuda-graph}"

printf "\n### displaying all environment variables ...\n"
printenv

printf "\n### starting SGLang ...\n"
SGL_COMMAND="python${PYTHON_VERSION} -m sglang.launch_server \
  --model ${SGL_MODEL} --model-path ${SGL_MODEL_DIR}/${SGL_MODEL} \
  --host ${SGL_HOST} --port ${SGL_PORT} --tensor-parallel-size $SGL_TP_SIZE \
  --log-level ${SGL_LOG_LEVEL} \
  ${SGL_PARAMS}"  \
  || sleep infinity
echo "sgl start command: $SGL_COMMAND"
eval "$SGL_COMMAND"