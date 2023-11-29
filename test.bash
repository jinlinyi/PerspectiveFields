#!/bin/bash

# Function to download a model if it doesn't already exist
download_model() {
    local model_url=$1
    local model_file_name=$2

    echo "Downloading model from URL: $model_url"
    if [ ! -f "models/${model_file_name}" ]; then
        wget -P models/ "${model_url}"
    fi
}

# Function to test a model
test_model() {
    local config_file=$1
    local model_file_name=$2
    local output_dir=${config_file%.yaml}

    echo "Testing model with config file: $config_file and weights: $model_file_name"
    python demo/demo.py \
    --config-file "${config_file}" \
    --input assets/imgs/cityscape.jpg \
    --output debug/"${output_dir}"/ \
    --opts MODEL.WEIGHTS "models/${model_file_name}"
}

# Download and test models
download_model "https://www.dropbox.com/s/z2dja70bgy007su/paramnet_360cities_edina_rpf.pth" "paramnet_360cities_edina_rpf.pth"
download_model "https://www.dropbox.com/s/nt29e1pi83mm1va/paramnet_360cities_edina_rpfpp.pth" "paramnet_360cities_edina_rpfpp.pth"
download_model "https://www.dropbox.com/s/czqrepqe7x70b7y/cvpr2023.pth" "cvpr2023.pth"
download_model "https://www.dropbox.com/s/ufdadxigewakzlz/paramnet_gsv_rpfpp.pth" "paramnet_gsv_rpfpp.pth"
download_model "https://www.dropbox.com/s/g6xwbgnkggapyeu/paramnet_gsv_rpf.pth" "paramnet_gsv_rpf.pth"

# Test a specific model (modify as needed)
# Test all models
test_model "models/paramnet_360cities_edina_rpf.yaml" "paramnet_360cities_edina_rpf.pth"
test_model "models/paramnet_360cities_edina_rpfpp.yaml" "paramnet_360cities_edina_rpfpp.pth"
test_model "models/cvpr2023.yaml" "cvpr2023.pth"
test_model "models/paramnet_gsv_rpfpp.yaml" "paramnet_gsv_rpfpp.pth"
test_model "models/paramnet_gsv_rpf.yaml" "paramnet_gsv_rpf.pth"