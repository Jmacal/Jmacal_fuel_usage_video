## Setup Instructions

1. Install Valhalla:
    ```bash
    mkdir custom_files
    wget -O custom_files/united-kingdom-latest.osm.pbf https://download.geofabrik.de/europe/united-kingdom-latest.osm.pbf
    docker run -dt -e server_threads=15 --name valhalla_gis-ops -p 8002:8002 -v $PWD/custom_files:/custom_files ghcr.io/gis-ops/docker-valhalla/valhalla:latest
    ```

2. Run Valhalla:
   ```bash
   docker start valhalla_gis-ops
   ```
   
3. Wait for installation to complete, check progress using:
   ```bash
   docker logs valhalla_gis-ops
   ```

4. Clone the repository:

    ```bash
    git clone https://github.com/BucherMunicipal/fuel_usage_video
    ```

5. Navigate to the project directory:

    ```bash
    cd fuel_usage_video
    ```

6. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

7. Open and run `fuel_usage_video_script.ipynb`


### Directory Structure

The following directories will be created:

- `mapping_data`
- `mapping_data/log_files`
- `mapping_data/csv_data`
- `mapping_data/format_csv_data`
- `mapping_data/map_frames`
- `mapping_data/graph_frames`
- `mapping_data/merged_frames`
- `mapping_data/created_videos`
- `mapping_data/videos`
- `mapping_data/video_frames`

### Preparing Input Data

- Drag and drop your `.log` sweep data files into the `mapping_data/log_files` directory.

### Generating Fuel Usage Videos

- Open and run `fuel_usage_video_script.ipynb` again
- A fuel usage video will be generated for each log file.

### Output

The generated videos will be saved in the `created_videos` directory.
