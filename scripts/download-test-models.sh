#!/bin/bash

cd whisper.cpp/models

./download-ggml-model.sh base
./download-vad-model.sh silero-v5.1.2
