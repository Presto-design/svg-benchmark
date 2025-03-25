#!/bin/bash
poetry run pytest \
    svg_benchmark/utils/svg_processing.py \
    svg_benchmark/utils/image_processing.py \
    svg_benchmark/utils/model_interface.py \
    svg_benchmark/models/test_presto_model.py \
    -v 