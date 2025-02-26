
import streamlit as st
import folium
from streamlit_folium import st_folium
import ollama

import os
import time
import tempfile
import zipfile
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from samgeo import tms_to_geotiff
import pyautogui
import requests
import numpy as np
from pyproj import Transformer
from pyproj import CRS
import math
from PIL import Image
import sys

st.header('🎈 Streamlit App for Spatial Data Analysis')

#--------------- Methods ---------------

class processData:
    def __init__(self, image=None, images=None, parcels=None, api_key=None):
        self.img = image
        self.imgs = images
        self.parcels = parcels
        self.key = api_key

    def loadParcelsAndMap(self, parcels=None, map=None, bbox=None):
        if parcels != None:
            # import shp data
            self.parcels = parcels
            # import map
            if map != None:
                m = rasterio.open(map)
            if bbox != None:
                m = rasterio.open(self.getMap(bbox))
    
    def getSV(self, centroid, epsg):
        bbox = self.projection(centroid, epsg)
        url = f"https://graph.mapillary.com/images?access_token={self.key}&fields=id,compass_angle,thumb_1024_url,geometry&bbox={bbox}&is_pano=true"
        response = requests.get(url).json()
        # find the closest image
        response = self.closest(centroid, response)
        # Extract Image ID, Compass Angle, image url, and coordinates
        img_id = response.iloc[0,0]
        img_heading = float(response.iloc[0,1])
        img_url = response.iloc[0,2]
        image_lon, image_lat = response.iloc[0,5]
        # calculate bearing to the house
        bearing_to_house = self.calculate_bearing(image_lat, image_lon, centroid.y, centroid.x)
        relative_heading = (bearing_to_house - img_heading) % 360
        # Download Image
        sv_data = requests.get(img_url).content
        with open("sv.jpg", "wb") as handler:
            handler.write(sv_data)
        # Load 360 Image
        image = Image.open("sv.jpg")
        width, height = image.size 
        # Convert relative heading to pixel offset
        crop_width = width // 4  # Approximate a 90-degree field of view
        center_x = int((relative_heading / 360) * width)
        # Crop Image
        left = max(0, center_x - crop_width // 2)
        right = min(width, center_x + crop_width // 2)
        cropped_image = image.crop((left, 0, right, height))
        return cropped_image
    
    def projection(self, centroid, epsg):
        x, y = self.degree2dis(centroid, epsg)
        
        # Get unit name (meters, degrees, etc.)
        crs = CRS.from_epsg(epsg)
        unit_name = crs.axis_info[0].unit_name
        # set search distance to 25 meters
        r = 25
        if unit_name == 'foot':
            r = 82.021
        elif unit_name == 'degree':
            print("Error: epsg must be projected system.")
            sys.exit(1)

        x_min = x - r
        y_min = y - r
        x_max = x + r
        y_max = y + r
        # Convert to EPSG:4326 (Lat/Lon) 
        x_min, y_min = self.dis2degree(x_min, y_min, epsg)
        x_max, y_max = self.dis2degree(x_max, y_max, epsg)
        return f'{x_min},{y_min},{x_max},{y_max}'

    def dis2degree(self, ptx, pty, epsg):
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        x, y = transformer.transform(ptx, pty)
        return x, y
    
    def degree2dis(self, pt, epsg):
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(pt.x, pt.y)
        return x, y
    
    def closest(self, centroid, response):
        c = [centroid.x, centroid.y]
        res_df = pd.DataFrame(response['data'])
        res_df[['point','coordinates']] = pd.DataFrame(res_df.geometry.tolist(), index= res_df.index)
        res_df[['lon','lat']] = pd.DataFrame(res_df.coordinates.tolist(), index= res_df.index)
        id_array = np.array(res_df['id'])
        lon_array = np.array(res_df['lon'])
        lat_array = np.array(res_df['lat'])
        dis_array = (lon_array-c[0])*(lon_array-c[0]) + (lat_array-c[1])*(lat_array-c[1])
        ind = np.where(dis_array == np.min(dis_array))[0]
        id = id_array[ind][0]
        return res_df.loc[res_df['id'] == id]
    
    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        delta_lon = lon2 - lon1

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360  # Normalize to 0-360
    
    def oneImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        return self.LLM_chat(system=system, prompt=prompt, img=self.img, 
                             temp=temp, top_k=top_k, top_p=top_p)
    
    def loopImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        for img in self.imgs:
            self.LLM_chat(system=system, prompt=prompt, img=img, 
                          temp=temp, top_k=top_k, top_p=top_p)
            
    def loopParcelChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        dic = {
            "lon": [],
            "lat": [],
            'response': [],
        }
        n = 0
        for i in range(len(self.parcels)):
            # Get the extent of one polygon from the filtered GeoDataFrame
            polygon = self.parcels.geometry.iloc[i]
            minx, miny, maxx, maxy = polygon.bounds
            bbox = [minx, miny, maxx, maxy]
            # Download data using tms_to_geotiff
            image = "parecel.tif"
            tms_to_geotiff(output=image, bbox=bbox, zoom=22, 
                           source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
                           overwrite=True)
            # Clip the image with the polygon
            with rasterio.open(image) as src:
                # Reproject the polygon back to match raster CRS
                polygon = self.parcels.to_crs(src.crs).geometry.iloc[i]
                out_image, out_transform = mask(src, [polygon], crop=True)
                out_meta = src.meta.copy()

            out_meta.update({
                "driver": "JPEG",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "count": 3 #Ensure RGB (3 bands)
            })

            clipped_image = 'clip.jpg'
            with rasterio.open(clipped_image, "w", **out_meta) as dest:
                dest.write(out_image)

            res = self.LLM_chat(system=system, 
                                prompt=prompt, 
                                img=clipped_image, 
                                temp=temp, 
                                top_k=top_k, 
                                top_p=top_p)
            dic['lon'].append(self.parcels.centroid.x.iloc[i])
            dic['lat'].append(self.parcels.centroid.y.iloc[i])
            dic['response'].append(res)
            progress_bar.progress((i+1)/len(self.parcels))
            # # show the image
            # with st.empty():
            #     st.image(clipped_image, caption="clipped parcel/block", width=200)
            #     st.write(res)
            #     time.sleep(1)
        return dic
    
    def LLM_chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
        if prompt != None and img != None:
            res = ollama.chat(
                model='llama3.2-vision',
                messages=[
                    {
                        'role': 'system',
                        'content': system
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [img]
                    }
                ],
                options={
                    "temperature":temp,
                    "top_k":top_k,
                    "top_p":top_p
                }
            )
            return res['message']['content']

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

@st.cache_data
def convert_df(data):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return data.to_csv().encode("utf-8")

#--------------- User Interface & functions ----------------

parcels_ = None
image_ = None
output = None

# parameters
para1, para2, para3 = st.columns(3)
with para1:
    tempr = st.slider("temperature", 0.0, 1.0, 0.5)
with para2:
    top_k = st.slider("top_k", 0.0, 1.0, 0.5)
with para3:
    top_p = st.slider("top_p", 0.0, 1.0, 0.5)

# prompt area
text1, text2 = st.columns(2)
with text1:
    system_info = st.text_area("Enter system context (optional):")
with text2:
    prompt = st.text_area("Enter your prompt (required):")

#------------------ tabs ------------------
tab_single_img_upload, tab_parcel_upload, tab_streetview, tab_parcel_streetview = st.tabs(
    [
        "Single Image", 
        "Parcel/Block(shp file)", 
        "Single Street View", 
        "Parcel & Street View"
    ]
)

with tab_single_img_upload:
    # buttons for uploading files
    img_uploader = st.file_uploader("Upload image data", type=["png", "jpg", "jpeg"])
    if img_uploader:
        image_ = img_uploader.read()
        st.write(f"You uploaded {img_uploader.name}")
        if image_:
            st.image(image_, caption="input image")

with tab_parcel_upload:
    # buttons for uploading files
    parcel_uploader = st.file_uploader("Upload parcel data", type=["zip"])
    if parcel_uploader:
        st.write(f"You uploaded {parcel_uploader.name}")
        if parcel_uploader:
            parcels_ = loadSHP(parcel_uploader)
            st.dataframe(parcels_)

with tab_streetview:
    # check box for street view
    mapillary_styles = {
        "Photo": "photo",
        "Split": "split",
        "Classic": "classic",
    }
    selected_style = st.selectbox("select Mapillary Style", list(mapillary_styles.keys()))

    # text input for street view link
    sv_link = st.text_input("street view link", "https://www.mapillary.com/app/?pKey=763349552242642&focus=photo")
    # extract key from the link
    img_key = extract_imgkey(sv_link)
    # Embed the entire Mapillary web app
    mapillary_embed_html = f"""
    <iframe 
        id="mapillarySection"
        width="100%" height="500" 
        src="https://www.mapillary.com/embed?map_style=Mapillary%20light&image_key={img_key}&x=0.5&y=0.5&style={mapillary_styles[selected_style]}"
        frameborder="0">
    </iframe>
    """
    # Embed
    st.components.v1.html(mapillary_embed_html, height=500)
    if img_key:
        st.write(f"Crrent Street View Image Key: {img_key}")

with tab_parcel_streetview:
    # text input for street view link
    sv_key = st.text_input("mapillary key (reqired)👇")
    # buttons for uploading files
    parcel_uploader_ = st.file_uploader("Upload parcel data_(required)", type=["zip"])
    if parcel_uploader_:
        st.write(f"You uploaded {parcel_uploader_.name}")
        if parcel_uploader_:
            parcels_ = loadSHP(parcel_uploader_)
            st.dataframe(parcels_)

#----------------- process data -----------------

# buttons for sending prompts/running model
with tab_single_img_upload:
    btn_send = st.button("process one image", 
                         key="button_send", 
                         help="click to send your prompt and image", 
                         type='secondary', disabled=False)
    if btn_send:
        inputData = processData(image=image_)
        if inputData != None:
            res = inputData.oneImgChat(prompt=prompt, temp=tempr, top_k=top_k, top_p=top_p)
            st.write(res)
with tab_parcel_upload:
    sample_on = st.toggle("randonm sample 2 rows for testing", True)
    btn1, btn2 = st.columns(2)
    with btn1:
        btn_send = st.button("process vector data", 
                            key="button_load_model", 
                            help="click to load model for all parcels", 
                            type='secondary', disabled=False)

    if sample_on:
        isSample = True
    else:
        isSample = False
    if btn_send:
        if isSample:
            random_sample = parcels_.sample(n=2)
            inputData = processData(parcels=random_sample)
        else:
            inputData = processData(parcels=parcels_)
        if inputData != None:
            res = inputData.loopParcelChat(prompt=prompt, temp=tempr, top_k=top_k, top_p=top_p)
            # convert to dataframe
            output = pd.DataFrame(res)

    if output is not None:
        st.dataframe(output)
        csv = convert_df(output)
        # download the result as csv
        with btn2:
            st.download_button(label="Download result as CSV", data=csv, file_name="output.csv", mime="text/csv")
        # plot on an interactive map
        # m = folium.Map(location=[output['lat'].mean(), output['lon'].mean()], zoom_start=12)
        # for i, row in output.iterrows():
        #     folium.Marker(
        #         location=[row['lat'], row['lon']],
        #         popup=row['response']
        #     ).add_to(m)
        # st_folium(m)
with tab_streetview:
    btn_send = st.button("process street view", 
                         key="button_send_steetview",  
                         type='secondary', disabled=False)
    if btn_send:
        st.components.v1.html(
            """
            <script>
            function scrollToIframe() {
                document.getElementById("mapillarySection").scrollIntoView({ behavior: 'smooth' });
            }
            scrollToIframe();
            </script>
            """,
            height=0,
        )
        screenshot_path = capture_screenshot()
        if screenshot_path:
            inputData = processData(image=screenshot_path)
            if inputData != None:
                sys_ = """
                Pretent there is only the street view image.
                you should only look at and talk about the built/natual environments 
                in the street view. You need to ignore and not talk about the user interface elements 
                (such as the mapillary logo and a green "Contribute Images" button)
                in streey view image
                """ + system_info
                res = inputData.oneImgChat(system=sys_, prompt=prompt, temp=tempr, top_k=top_k, top_p=top_p)
                st.write(res)
