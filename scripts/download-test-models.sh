#!/bin/bash

cd whisper.cpp/models

./download-ggml-model.sh tiny.en
./download-vad-model.sh silero-v5.1.2
