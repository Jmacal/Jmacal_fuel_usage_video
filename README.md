Install Valhalla & create docker container

https://github.com/gis-ops/docker-valhalla

command to run docker 

```
docker run -dt -e server_threads=15 --name valhalla_gis-ops -p 8002:8002  -v $PWD/custom_files:/custom_files ghcr.io/gis-ops/docker-valhalla/valhalla:latest
```


the output videos will be at 


```
./fuel_usage_video/mapping_data/created_videos/
```
