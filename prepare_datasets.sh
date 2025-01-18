#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Create a directory for datasets
mkdir -p datasets
cd datasets

download_and_extract() {
    local url=$1
    local zip_file=$2
    local dataset_number=$3

    echo "Downloading dataset $dataset_number..."
    if curl -L "$url" > "$zip_file"; then
        echo "Download complete. Extracting..."
        if unzip -o "$zip_file"; then
            echo "Extraction complete."
            
            # Move and rename the data.yaml file
            if [ -f "data.yaml" ]; then
                mv "data.yaml" "data_$dataset_number.yaml"
                echo "Renamed data.yaml to data_$dataset_number.yaml"
            else
                echo "Warning: data.yaml not found in dataset $dataset_number"
            fi
            
            # Merge train, valid, and test directories
            for dir in train valid test; do
                mkdir -p "../$dir"
                if [ -d "$dir" ]; then
                    echo "Moving $dir files for dataset $dataset_number"
                    mv "$dir"/* "../$dir/" 2>/dev/null || true
                else
                    echo "Warning: $dir directory not found in dataset $dataset_number"
                fi
            done
            
            # Clean up
            rm "$zip_file"
            echo "Cleaned up temporary files for dataset $dataset_number"
        else
            echo "Error: Failed to extract $zip_file"
            return 1
        fi
    else
        echo "Error: Failed to download dataset $dataset_number"
        return 1
    fi
}

# Replace these URLs with your own Roboflow dataset URLs
# Format: https://universe.roboflow.com/ds/XXXXX?key=YYYYY
download_and_extract "https://universe.roboflow.com/ds/fDIqqNh4e5?key=L3vxsXJjjy" "roboflow1.zip" 1
download_and_extract "https://universe.roboflow.com/ds/kZJdgr4515?key=yogetoOjSg" "roboflow2.zip" 2
download_and_extract "https://universe.roboflow.com/ds/84rwQwVln7?key=4YCb3Tk4Ad" "roboflow3.zip" 3
download_and_extract "https://universe.roboflow.com/ds/A5dpCQbRRU?key=m1K7DCWufP" "roboflow4.zip" 4
download_and_extract "https://universe.roboflow.com/ds/RmwavhOPcb?key=NeEfEQhkmr" "roboflow5.zip" 5
download_and_extract "https://universe.roboflow.com/ds/rzjXiZnRQM?key=101S2sjvkD" "roboflow6.zip" 6
download_and_extract "https://universe.roboflow.com/ds/ouVtRybOAT?key=5dQkBdGlmN" "roboflow7.zip" 7
download_and_extract "https://universe.roboflow.com/ds/24PuMLATKb?key=ieB1H9NTQc" "roboflow8.zip" 8
download_and_extract "https://universe.roboflow.com/ds/NT61vZE4wn?key=sLJZjrfvns" "roboflow9.zip" 9
download_and_extract "https://universe.roboflow.com/ds/FTGbRNVpXd?key=NaTwGsx4DG" "roboflow10.zip" 10
download_and_extract "https://universe.roboflow.com/ds/8aqZNeBV8D?key=RFUCMPQQqv" "roboflow11.zip" 11
download_and_extract "https://universe.roboflow.com/ds/x1ff3Ks2tC?key=HTvO39r2wc" "roboflow12.zip" 12
download_and_extract "https://universe.roboflow.com/ds/FiNIfsiTyH?key=qv8CyqwLgE" "roboflow13.zip" 13
download_and_extract "https://universe.roboflow.com/ds/fF9Y9eTRI8?key=oGt3L0SgVv" "roboflow14.zip" 14
download_and_extract "https://universe.roboflow.com/ds/3FmmkGaHu7?key=iMwvcKcSIb" "roboflow15.zip" 15
download_and_extract "https://universe.roboflow.com/ds/GsEBWqWDBg?key=krldymL6Al" "roboflow16.zip" 16
download_and_extract "https://universe.roboflow.com/ds/FrxvEfaDQL?key=HHWWHOYee0" "roboflow17.zip" 17
download_and_extract "https://universe.roboflow.com/ds/m1utjRX8zo?key=LXCZTvisJ2" "roboflow18.zip" 18
download_and_extract "https://universe.roboflow.com/ds/uQaAPiaxCR?key=EummVCRzZI" "roboflow19.zip" 19
download_and_extract "https://universe.roboflow.com/ds/cMIOmHAVfH?key=7o4keVNdPX" "roboflow20.zip" 20
download_and_extract "https://universe.roboflow.com/ds/vrs8SH1dZr?key=eUdOWuqX7C" "roboflow21.zip" 21
download_and_extract "https://universe.roboflow.com/ds/30VvONkRTm?key=huSsseBSKO" "roboflow22.zip" 22
download_and_extract "https://universe.roboflow.com/ds/vhqBruhUfA?key=ELQdzRR2vs" "roboflow23.zip" 23
download_and_extract "https://universe.roboflow.com/ds/YWOsWi1NNB?key=JNO7Bs9yAj" "roboflow24.zip" 24
download_and_extract "https://universe.roboflow.com/ds/hdEI3h0SmG?key=3USzuo4hYN" "roboflow25.zip" 25
download_and_extract "https://universe.roboflow.com/ds/N3fl1TVrBd?key=0E46AZcju2" "roboflow26.zip" 26



echo "All datasets downloaded and extracted successfully."

# Count total images
total_images=$(find . -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
echo "Total number of images: $total_images"
