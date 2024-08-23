#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import folium
from folium import plugins
import numpy as np
from folium import LayerControl as folium_LayerControl
import csv
import requests
import json
import shutil
from datetime import datetime
import time
import imgkit
from selenium import webdriver
import moviepy
from moviepy.editor import ImageSequenceClip
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
from moviepy.editor import clips_array
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import mercantile


def sort_sweep_sessions(csv_file_dir):
    for file in os.listdir(csv_file_dir):
        if os.path.isdir(os.path.join(csv_file_dir, file)):
            continue
        date = '_'.join(file.split('_')[1:4])
        if not os.path.isdir(os.path.join(csv_file_dir, date)):
            os.makedirs(os.path.join(csv_file_dir, date))
        
        shutil.move(os.path.join(csv_file_dir, file), os.path.join(csv_file_dir, date))
        
    print("Successfully sorted log files into dates")


def data_to_csv(input_data_dir, output_csv_dir):

    try:
        for file in os.listdir(input_data_dir):

            if not os.path.exists(os.path.join(input_data_dir, file)):
                print(f"File {file} does not exist.")
                continue
            elif os.path.exists(os.path.join(output_csv_dir, file[:-4]+'.csv')):
                print(f"CSV {file[:-4]+'.csv'} already exists.")
                continue
            else:
                columns = []
                data = []
                reading_data = False
        
                with open(os.path.join(input_data_dir, file), 'r') as data_file:
                    for row in data_file:
                        if row.split(' ')[0] == 'time' or row.split(' ')[0] == 'sats':
                            columns = row.split(' ')
                        elif row.split(']')[0] == '[data':
                            reading_data = True
                            continue
        
                        if reading_data:
                            data_row = row.split(' ')
                            data.append(data_row)
        
                with open(os.path.join(output_csv_dir, file[:-4]+'.csv'), 'w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerow(columns)
                    for d_r in data:
                        csv_writer.writerow(d_r)

        print("Data files have been successfully converted to csv")
        return True
        
    except Exception as e:
        print(e)
        return False



def format_data_files(csv_data_dir, formatted_csv_path):

    try:
        for csv_file in os.listdir(csv_data_dir):

            if not os.path.exists(os.path.join(csv_data_dir, csv_file)):
                print(f"File {csv_file} does not exist.")
                continue
            elif os.path.exists(os.path.join(formatted_csv_path, csv_file)) or os.path.exists(os.path.join(formatted_csv_path, '_'.join(csv_file.split('_')[1:4]), csv_file)):
                print(f"File {csv_file} already formatted.")
                continue
            else:

                date = '-'.join(csv_file.split('.')[0].split('_')[1:4])
                
                df = pd.read_csv(os.path.join(csv_data_dir, csv_file))
            
                df['DateTime'] = date + ' ' + df['time']
            
                df.to_csv(os.path.join(formatted_csv_path, csv_file), index=False)

        print("Csv files have been successfully formatted")
        return  True

    except Exception as e:
        print(e)
        return False


def get_data_values(formatted_csv_path):
    try:
        # Read the CSV file directly into a DataFrame
        df = pd.read_csv(formatted_csv_path)

        # Extract the required columns directly into numpy arrays
        long_data = df['Longitude'].to_numpy()
        lat_data = df['Latitude'].to_numpy()
        
        nozzle_down1_data = df['Nozzle1downTMSCS'].to_numpy()
        nozzle_down2_data = df['Nozzle2downTMS'].to_numpy()
        fuel_rate_data = df['EngineFuelRateTMSCS'].to_numpy()

        # Convert datetime strings to datetime objects
        datetime_data = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')

        print("Data has been successfully collated")
        return long_data, lat_data, datetime_data, fuel_rate_data, nozzle_down1_data, nozzle_down2_data

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None, None, None, None, None, None, None


def snap_to_road(lat_vals, lon_vals):
    url = 'http://localhost:8002/trace_attributes'
    headers = {
        'Content-Type': 'application/json',
    }

    shape_arr = [{"lat": lat, "lon": lon} for lat, lon in zip(lat_vals, lon_vals)]

    data = {
        "shape": shape_arr,
        "costing": "auto",
        "shape_match": "map_snap"
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_json = response.json()

        snapped_lat_vals = [point["lat"] for point in response_json["matched_points"]]
        snapped_lon_vals = [point["lon"] for point in response_json["matched_points"]]

        return snapped_lat_vals, snapped_lon_vals
        
    else:
        print(f"Error: {response.status_code} {response}")


def rgb_to_hex(r, g, b):
    """Convert RGB color to Hex color."""
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def get_color(x):
    if x < 0.3:
        # Transition from green to yellow
        red = int(x / 0.3 * 255)
        green = 255
    elif x < 0.6:
        # Transition from yellow to orange
        red = 255
        green = int((1 - (x - 0.3) / 0.3) * 255)
    else:
        # Transition from orange to red
        red = 255
        green = int((1 - (x - 0.6) / 0.4) * 165) # Adjusting for a more distinct orange
    blue = 0 # Blue remains 0 throughout the transition
    return [red, green, blue]


def get_fuel_rate_color(fuel_rate, min_val, max_val):

    # Normalize fuel_rate
    normalized_fuel_rate = (fuel_rate - min_val) / (max_val - min_val)
    # Get color based on normalized fuel_rate
    color = get_color(normalized_fuel_rate)
    # Convert color to hexadecimal representation
    color = rgb_to_hex(color[0], color[1], color[2])

    return color


def correct_lon_lat(long_values, lat_values):
    if len(lat_values) < 16000:
        lat_values, long_values = snap_to_road(lat_values, long_values)
    else:
        i = 0
        while i < len(lat_values):
            segment_length = min(16000, len(lat_values) - i)
            segment_lat, segment_long = snap_to_road(lat_values[i:i+segment_length], long_values[i:i+segment_length])
            lat_values[i:i+segment_length] = segment_lat
            long_values[i:i+segment_length] = segment_long
            i += segment_length

    return long_values, lat_values


def html_to_png(html_file_path, output_png_path):
    # Set up Chrome WebDriver options
    options = Options()
    options.add_argument('--headless')  # Run Chrome in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU acceleration (may speed up rendering)
    options.add_argument('--no-sandbox')  # Disable sandbox for compatibility with Docker

    # Create WebDriver using a context manager
    with webdriver.Chrome(options=options) as driver:
        try:
            # Load the HTML file
            driver.get('file://' + html_file_path)

            # Wait for specific element to be visible (adjust according to your HTML content)
            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))

            # Capture screenshot and save as PNG
            driver.save_screenshot(output_png_path)

        except Exception as e:
            print(f"Error occurred: {e}")


def create_video_from_images(image_folder, output_video, fps):
    image_files = [f"frame_{n}.png" for n in range(len(os.listdir(image_folder)))]

    # Open the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)
            video.write(frame)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Release resources
        cv2.destroyAllWindows()
        video.release()


def calculate_ground_speed(lon_lat_t, prev_lon_lat_t):
    # Validate input values to ensure they are within feasible and safe ranges
    delta_lat = abs(float(lon_lat_t[1]) - float(prev_lon_lat_t[1]))
    delta_lon = abs(float(lon_lat_t[0]) - float(prev_lon_lat_t[0]))
    delta_t = abs((lon_lat_t[2] - prev_lon_lat_t[2]).total_seconds())

    if not (-180 <= delta_lat <= 180 and -180 <= delta_lon <= 180):
        raise ValueError("Latitude and longitude change must be between -180 and 180 degrees.")
    if delta_t <= 0:
        raise ValueError("Time interval must be greater than zero.")

    # Earth's radius in kilometers (mean radius)
    R = 6371.0

    # Convert latitude and longitude from degrees to radians for calculation
    delta_lat_rad = math.radians(delta_lat)
    delta_lon_rad = math.radians(delta_lon)

    # Equirectangular approximation calculation
    x = delta_lon_rad * math.cos(math.radians((float(lon_lat_t[1]) + float(prev_lon_lat_t[1])) / 2.0))
    y = delta_lat_rad
    d = R * math.sqrt(x ** 2 + y ** 2) * 1000  # Distance in meters between the two points

    # Speed calculation
    speed_mps = d / delta_t

    # Conversion to other units
    speed_mph = speed_mps * 2.23694  # Conversion to miles per hour
    speed_kph = speed_mps * 3.6      # Conversion to kilometers per hour

    return speed_mps, speed_mph, speed_kph


def get_velocity_data(lon_data, lat_data, time_data):
    
    velocity_data = [0.0]  # Initial velocity
    
    
    for i in range(1, len(time_data)):

        if i >= 10:
            prev_lon, prev_lat, prev_time = lon_data[i-10], lat_data[i-10], time_data[i-10]
        else:
            prev_lon, prev_lat, prev_time = lon_data[0], lat_data[0], time_data[0]
        
        # Get the current row of status information
        lon, lat, curr_time = lon_data[i], lat_data[i], time_data[i]

        # Calculate time difference
        time_diff = (curr_time - prev_time).total_seconds()

        if time_diff == 0:
            # If time difference is zero, maintain previous velocity
            velocity_data.append(velocity_data[-1])
        else:
            # Calculate ground speed
            speed_mps, speed_mph, speed_kph = calculate_ground_speed([lon, lat, curr_time], [prev_lon, prev_lat, prev_time])
            velocity_data.append(speed_mph)

    return velocity_data


def create_graph_images(velocity_data, fuel_rate_data, graph_frames_dir, max_frames):

    frame_numbers = np.arange(len(fuel_rate_data))

    try:

        # Create figure and subplots
        fig, ax1 = plt.subplots()

        # Plot fuel rate on the first subplot
        ax1.plot(frame_numbers, fuel_rate_data, color='red')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Fuel Rate', color='red')

        # Create a second y-axis for velocity and plot it on the same subplot
        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, velocity_data, color='blue')
        ax2.set_ylabel('Velocity (mph)', color='blue')
        
        
        for frame_number in frame_numbers:
            
            # Draw a vertical line at the current frame number
            frame_line = ax1.axvline(x=frame_number, color='green', linestyle='--')

            # Save the plot as a numbered PNG file
            plt.savefig(os.path.join(graph_frames_dir, f'frame_{frame_number}.png'))

            frame_line.remove()

            if frame_number == max_frames:
                break

        # Close the plot to release memory
        plt.close(fig)

    except Exception as e:
        print(f"Error occurred: {e}")
        # Ensure plots are closed in case of exception
        plt.close('all')


def merge_frames(map_frame_dir, graph_frame_dir, video_frame_dir, merged_frame_dir, num_of_frames, electric):
    for n in range(num_of_frames):
        with Image.open(os.path.join(map_frame_dir, f'frame_{n}.png')) as map_frame, \
             Image.open(os.path.join(graph_frame_dir, f'frame_{n}.png')).resize(map_frame.size) as graph_frame:
 
                if electric:
                    video_frame = Image.open(os.path.join(video_frame_dir, 'left', f'frame_{n}.png')).resize((map_frame.width*2, map_frame.height*2)) 
    
                    merged_frame = Image.new('RGB', (map_frame.width*3, map_frame.height*2))
        
                    merged_frame.paste(graph_frame, (map_frame.width*2, 0))
                    merged_frame.paste(map_frame, (map_frame.width*2, map_frame.height))
        
                    merged_frame.paste(video_frame, (0, 0))
                
                else:
                    cam_pos = ['back', 'right', 'front', 'left']
                    video_frames = [Image.open(os.path.join(video_frame_dir, vid_dir, f'frame_{n}.png')).resize(map_frame.size) 
                                    for vid_dir in cam_pos]
        
                    merged_frame = Image.new('RGB', (map_frame.width*3, map_frame.height*2))
        
                    merged_frame.paste(graph_frame, (map_frame.width*2, 0))
                    merged_frame.paste(map_frame, (map_frame.width*2, map_frame.height))
        
                    merged_frame.paste(video_frames[0], (map_frame.width, 0))
                    merged_frame.paste(video_frames[1], (map_frame.width, map_frame.height))
                    merged_frame.paste(video_frames[2], (0, 0))
                    merged_frame.paste(video_frames[3].rotate(90, expand=False), (0, map_frame.height))
    
                # Save the merged image
                merged_frame.save(os.path.join(merged_frame_dir, f"frame_{n}.png"))


    print("Frames merged")



def extract_frames(video_path, output_dir, target_fps):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get original frame rate
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame skip interval
    frame_skip_interval = int(round(original_fps / target_fps))

    # Get the total number of frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0

    for n in range(total_frames):

        # Read next frame
        success, frame = video_capture.read()

        # Save frame as PNG image if it's within the frame skip interval
        if n % frame_skip_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_count}.png", frame)
            frame_count += 1

    # Release the video capture object
    video_capture.release()



def add_info_to_frame(frame_path, frame_n, electric):

    frame = Image.open(frame_path)

    w, h = frame.size

    frame_with_info = np.array(frame.copy())

    font = cv2.FONT_HERSHEY_SIMPLEX  # Set font style
    font_scale = 1  # Set font scale

    # Loop twice for two different text styles (black and white)
    for a in range(2):
        # Set text style for the first iteration
        if a == 0:
            colours = (0, 0, 0)  # black
            font_thickness = 12
        # Set text style for the second iteration
        else:
            colours = (255, 255, 255)  # white
            font_thickness = 3

        if not electric:
            cv2.putText(frame_with_info, f'Frame {frame_n}', (((w//3)+20), 40), font, font_scale, colours, font_thickness)
            cv2.putText(frame_with_info, f'Frame {frame_n}', (20, ((h//2)+40)), font, font_scale, colours, font_thickness)
            cv2.putText(frame_with_info, f'Frame {frame_n}', (((w//3)+20), ((h//2)+40)), font, font_scale, colours, font_thickness)

            cv2.putText(frame_with_info, f'Right', (((w//3)+20), (h-20)), font, font_scale, colours, font_thickness)
            cv2.putText(frame_with_info, f'Front', (20, ((h//2)-20)), font, font_scale, colours, font_thickness)
            cv2.putText(frame_with_info, f'Back', (((w//3)+20), ((h//2)-20)), font, font_scale, colours, font_thickness)

        cv2.putText(frame_with_info, f'Frame {frame_n}', (20, 40), font, font_scale, colours, font_thickness)
    
        cv2.putText(frame_with_info, f'Left', (20, (h-20)), font, font_scale, colours, font_thickness)
        

    os.remove(frame_path)
    Image.fromarray(frame_with_info).save(frame_path)
    

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 1 << zoom
  xtile = (lon_deg + 180.0) / 360.0 * n
  ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
  return xtile, ytile

def num2pix(xtile, ytile):
    pixel_x = (xtile % 1.0) * 256
    pixel_y = (ytile % 1.0) * 256
    return [pixel_x, pixel_y]
    

def create_map_frames(map, lat_data, long_data, map_frame_dir, zoom, fuel_rate_data, map_image_dir):

    frame_n = 0

    for n in range(len(long_data)-1):

        tile_1 = deg2num(lat_data[n], long_data[n], zoom)
        tile_2 = deg2num(lat_data[n+1], long_data[n+1], zoom)

        if n == 0:
            map.location = [lat_data[n], long_data[n]]

            # Creates html from drawn map and converts the html to a frame png deleting the html file
            map.save(f"{map_image_dir}/map.html")
            html_to_png(f"{map_image_dir}/map.html", f"{map_image_dir}/map.png")    
            os.remove(f"{map_image_dir}/map.html")
            map_png = Image.open(f"{map_image_dir}/map.png")
            w, h = map_png.size
            draw = ImageDraw.Draw(map_png)

            location_tile = deg2num(lat_data[n], long_data[n], zoom)


        if int(tile_1[0]) != int(location_tile[0]) or int(tile_1[1]) != int(location_tile[1]) or int(tile_2[0]) != int(location_tile[0]) or int(tile_2[1]) != int(location_tile[1]):
            map.location = [lat_data[n], long_data[n]]

            # Creates html from drawn map and converts the html to a frame png deleting the html file
            map.save(f"{map_image_dir}/map.html")
            html_to_png(f"{map_image_dir}/map.html", f"{map_image_dir}/map.png")    
            os.remove(f"{map_image_dir}/map.html")
            map_png = Image.open(f"{map_image_dir}/map.png")
            w, h = map_png.size
            draw = ImageDraw.Draw(map_png)

            location_tile = deg2num(lat_data[n], long_data[n], zoom)

        if [lat_data[n], long_data[n]] == [lat_data[n+1], long_data[n+1]]:
            continue

        point_1 = num2pix(tile_1[0], tile_1[1])
        point_2 = num2pix(tile_2[0], tile_2[1])

        print(tile_1, tile_2)

        draw.line([(point_1[0], point_1[1]), (point_2[0], point_2[1])], fill=get_fuel_rate_color(fuel_rate_data[n], min(fuel_rate_data), max(fuel_rate_data)), width=3)
        map_png.save(f"{map_frame_dir}/frame_{frame_n}.png")
        frame_n += 1


if __name__ == "__main__":

    # Define directories
    base_dir = '/home/james/workspace/fuel_usage_video/mapping_data'
    data_dir = os.path.join(base_dir, 'log_files')
    csv_dir = os.path.join(base_dir, 'csv_data')
    format_csv_dir = os.path.join(base_dir, 'format_csv_data')
    map_frame_dir = os.path.join(base_dir, 'map_frames')
    graph_frame_dir = os.path.join(base_dir, 'graph_frames')
    merged_frame_dir = os.path.join(base_dir, 'merged_frames')
    output_videos_dir = os.path.join(base_dir, 'created_videos')
    sweeping_videos_dir = os.path.join(base_dir, 'videos')
    video_frame_dir = os.path.join(base_dir, 'video_frames')
    map_image_dir = os.path.join(base_dir, 'map_image')
    max_frames = 999999

    # Delete and recreate necessary directories
    for directory in [map_frame_dir, graph_frame_dir, merged_frame_dir, video_frame_dir]:
        shutil.rmtree(directory, ignore_errors=True)
        create_directory_if_not_exists(directory)


    # Convert log to csv
    if data_to_csv(data_dir, csv_dir) == False:
        exit()

    # Add additional columns to csv
    if format_data_files(csv_dir, format_csv_dir) == False:
        exit()

    # Iterate through each formatted csv
    for format_csv in os.listdir(format_csv_dir):
        
        # Extract data values from csv
        long_data, lat_data, datetime_data, fuel_rate_data, nozzle_down1_data, nozzle_down2_data = get_data_values(os.path.join(format_csv_dir, format_csv))

        # Calculate vehicle velocities
        velocity_data = get_velocity_data(long_data, lat_data, datetime_data)

        # Draw graph frames using velocity and fuel rate
        #create_graph_images(velocity_data, fuel_rate_data, graph_frame_dir, max_frames)
        
        # Use Valhalla API to snap lon lat values to road
        long_data, lat_data = correct_lon_lat(long_data, lat_data)
                                                                                                                   
        plot_data = []

        # Create open street map 
        map  = folium.Map(
                location=[np.array(lat_data).mean(),np.array(long_data).mean()],
                tiles='openstreetmap',
                zoom_start=18,
                prefer_canvas=True,
                zoom_control=False,
            )

        # Checks if the map frame directory exists
        if os.path.isdir(os.path.join(map_frame_dir, format_csv)[:-4]):
            continue

        create_map_frames(map, lat_data, long_data, map_frame_dir, 18, fuel_rate_data, map_image_dir)

        if os.path.exists(f"{map_frame_dir}/frame_{n}.png"):
            continue
        

        # Iterate through video
        for video_path in os.listdir(os.path.join(sweeping_videos_dir, format_csv[:-4])):
            
            # Vary process depending on if video is from electric truck or not
            if video_path.split('.')[-1] != 'h264':
                output_vid_frame_dir = os.path.join(video_frame_dir, video_path.split('_')[0])
                electric = False
            else:
                output_vid_frame_dir = os.path.join(video_frame_dir, 'left')
                electric = True
    
            # Split videos into frame images
            extract_frames(os.path.join(sweeping_videos_dir, format_csv[:-4], video_path), output_vid_frame_dir, 10)

        # Video should be created from minimun number of frames to avoid issues with varying number of frames
        num_of_frames = min(max_frames, len(os.listdir(graph_frame_dir)), len(os.listdir(map_frame_dir)), len(os.listdir(os.path.join(video_frame_dir, 'left'))), len(long_data))

        # Combine all frame images into one merged frame
        merge_frames(map_frame_dir, graph_frame_dir, video_frame_dir, merged_frame_dir, num_of_frames, electric)

        # Add information such as frame number to each frame
        for frame in os.listdir(merged_frame_dir):
            add_info_to_frame(os.path.join(merged_frame_dir, frame), int(frame.split('_')[1].split('.')[0]), electric)

        # Translate frames into final video
        create_video_from_images(merged_frame_dir, os.path.join(output_videos_dir, f"{format_csv[:-4]}.mp4"), fps=10)
        
    



