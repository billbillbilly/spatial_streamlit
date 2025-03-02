import os
import tempfile
import zipfile
import streamlit as st
import geopandas as gpd
import pyautogui

def loadSHP(file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save and extract ZIP file
        zip_path = os.path.join(tmp_dir, "uploaded_shapefile.zip")
        with open(zip_path, "wb") as f:
            f.write(file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find the .shp file, skipping __MACOSX directory and hidden files
        shp_file = None
        for root, dirs, files in os.walk(tmp_dir):
            if "__MACOSX" in root:
                continue
            for file in files:
                if file.endswith(".shp") and not file.startswith("._"):
                    shp_file = os.path.join(root, file)
                    break

        if shp_file:
            try:
                # Read shapefile
                gdf = gpd.read_file(shp_file)

                # Ensure CRS is WGS84 for visualization
                gdf = gdf.to_crs("EPSG:4326")

                # Convert datetime columns to strings
                for column in gdf.select_dtypes(include=["datetime", "datetime64[ns]"]).columns:
                    gdf[column] = gdf[column].astype(str)
                return gdf

            except Exception as e:
                st.error(f"Error reading or displaying Shapefile: {e}")
        else:
            st.error("No valid .shp file found in the uploaded ZIP.") 

# extract streetview key
def extract_imgkey(link):
    # extract key from the link
    key = link.split("pKey=")[1].split("&")[0]
    return key

# Function to take a screenshot
def capture_screenshot():
    screen_width, screen_height = pyautogui.size()
    screenshot = pyautogui.screenshot()  # Takes a full screenshot
    screenshot_path = "mapillary_screenshot.png"
    # Crop the screenshot based on screen_width and screen_height 
    cropped_image = screenshot.crop((int(screen_width*0.3),     # left
                                     int(screen_height*0.3),    # top
                                     int(screen_width*0.7),     # right
                                     int(screen_height*0.9)))   # bottom
    cropped_image.save(screenshot_path)
    # screenshot.save(screenshot_path)
    return screenshot_path