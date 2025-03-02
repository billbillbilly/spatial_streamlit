
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

from src.utils import loadSHP, extract_imgkey, capture_screenshot
from src.processData import processData

st.header('WebUI for urban envtionment inspection')

#--------------- Methods ---------------

# class processData:
#     def __init__(self, image=None, images=None, parcels=None):
#         self.img = image
#         self.imgs = images
#         self.parcels = parcels
    
#     def getSV(self, centroid, epsg, key):
#         bbox = self.projection(centroid, epsg)
#         url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_1024_url,geometry&bbox={bbox}&is_pano=true"
#         # response = None
#         # while not response or 'data' not in response:
#         response = requests.get(url).json()

#         # find the closest image
#         response = self.closest(centroid, response)
#         # Extract Image ID, Compass Angle, image url, and coordinates
#         img_id = response.iloc[0,0]
#         img_heading = float(response.iloc[0,1])
#         img_url = response.iloc[0,2]
#         image_lon, image_lat = response.iloc[0,5]
#         # calculate bearing to the house
#         bearing_to_house = self.calculate_bearing(image_lat, image_lon, centroid.y, centroid.x)
#         relative_heading = (bearing_to_house - img_heading) % 360
#         # Download Image
#         sv_data = requests.get(img_url).content
#         with open("sv.jpg", "wb") as handler:
#             handler.write(sv_data)
#         # Load 360 Image
#         image = Image.open("sv.jpg")
#         width, height = image.size 
#         # Convert relative heading to pixel offset
#         crop_width = width // 4  # Approximate a 90-degree field of view
#         center_x = int((relative_heading / 360) * width)
#         # Crop Image
#         left = max(0, center_x - crop_width // 2)
#         right = min(width, center_x + crop_width // 2)
#         cropped_image = image.crop((left, 0, right, height))
#         cropped_image.save("cropped_sv.jpg", format="JPEG")
#         print('street view downloaded')
#         return "cropped_sv.jpg"
    
#     def projection(self, centroid, epsg):
#         x, y = self.degree2dis(centroid, epsg)
#         # Get unit name (meters, degrees, etc.)
#         crs = CRS.from_epsg(epsg)
#         unit_name = crs.axis_info[0].unit_name
#         # set search distance to 25 meters
#         r = 50
#         if unit_name == 'foot':
#             r = 164.042
#         elif unit_name == 'degree':
#             print("Error: epsg must be projected system.")
#             sys.exit(1)
#         # set bbox
#         x_min = x - r
#         y_min = y - r
#         x_max = x + r
#         y_max = y + r
#         # Convert to EPSG:4326 (Lat/Lon) 
#         x_min, y_min = self.dis2degree(x_min, y_min, epsg)
#         x_max, y_max = self.dis2degree(x_max, y_max, epsg)
#         return f'{x_min},{y_min},{x_max},{y_max}'

#     def dis2degree(self, ptx, pty, epsg):
#         transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
#         x, y = transformer.transform(ptx, pty)
#         return x, y
    
#     def degree2dis(self, pt, epsg):
#         transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
#         x, y = transformer.transform(pt.x, pt.y)
#         return x, y
    
#     def closest(self, centroid, response):
#         c = [centroid.x, centroid.y]
#         res_df = pd.DataFrame(response['data'])
#         res_df[['point','coordinates']] = pd.DataFrame(res_df.geometry.tolist(), index= res_df.index)
#         res_df[['lon','lat']] = pd.DataFrame(res_df.coordinates.tolist(), index= res_df.index)
#         id_array = np.array(res_df['id'])
#         lon_array = np.array(res_df['lon'])
#         lat_array = np.array(res_df['lat'])
#         dis_array = (lon_array-c[0])*(lon_array-c[0]) + (lat_array-c[1])*(lat_array-c[1])
#         ind = np.where(dis_array == np.min(dis_array))[0]
#         id = id_array[ind][0]
#         return res_df.loc[res_df['id'] == id]
    
#     def calculate_bearing(self, lat1, lon1, lat2, lon2):
#         lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
#         delta_lon = lon2 - lon1

#         x = math.sin(delta_lon) * math.cos(lat2)
#         y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

#         bearing = math.degrees(math.atan2(x, y))
#         return (bearing + 360) % 360  # Normalize to 0-360
    
#     def oneImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
#         return self.LLM_chat(system=system, prompt=prompt, img=[self.img], 
#                              temp=temp, top_k=top_k, top_p=top_p)
    
#     def loopImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
#         for img in self.imgs:
#             self.LLM_chat(system=system, prompt=prompt, img=[img], 
#                           temp=temp, top_k=top_k, top_p=top_p)
            
#     def loopParcelChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None, withSV=False, epsg=None, key=None):
#         progress_text = "Operation in progress. Please wait."
#         progress_bar = st.progress(0, text=progress_text)
#         dic = {
#             "lon": [],
#             "lat": [],
#             'response': [],
#         }
#         n = 0
#         for i in range(len(self.parcels)):
#             # Get the extent of one polygon from the filtered GeoDataFrame
#             polygon = self.parcels.geometry.iloc[i]
#             centroid = polygon.centroid
#             minx, miny, maxx, maxy = polygon.bounds
#             bbox = [minx, miny, maxx, maxy]
#             # Download data using tms_to_geotiff
#             image = "parecel.tif"
#             tms_to_geotiff(output=image, bbox=bbox, zoom=22, 
#                            source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
#                            overwrite=True)
#             # Clip the image with the polygon
#             with rasterio.open(image) as src:
#                 # Reproject the polygon back to match raster CRS
#                 polygon = self.parcels.to_crs(src.crs).geometry.iloc[i]
#                 out_image, out_transform = mask(src, [polygon], crop=True)
#                 out_meta = src.meta.copy()

#             out_meta.update({
#                 "driver": "JPEG",
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform,
#                 "count": 3 #Ensure RGB (3 bands)
#             })

#             clipped_image = 'clip.jpg'
#             with rasterio.open(clipped_image, "w", **out_meta) as dest:
#                 dest.write(out_image)

#             # add images
#             input_imgs = [clipped_image]

#             if withSV == True and epsg != None and key != None:
#                 input_imgs += [self.getSV(centroid, epsg, key)]

#             res = self.LLM_chat(system=system, 
#                                 prompt=prompt, 
#                                 img=input_imgs, 
#                                 temp=temp, 
#                                 top_k=top_k, 
#                                 top_p=top_p)
#             dic['lon'].append(self.parcels.centroid.x.iloc[i])
#             dic['lat'].append(self.parcels.centroid.y.iloc[i])
#             dic['response'].append(res)
#             progress_bar.progress((i+1)/len(self.parcels))
#             # # show the image
#             # with st.empty():
#             #     st.image(clipped_image, caption="clipped parcel/block", width=200)
#             #     st.write(res)
#             #     time.sleep(1)
#         return dic
    
#     def LLM_chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
#         if prompt != None and img != None:
#             if len(img) == 1:
#                 return self.chat(system, prompt, img[0], temp, top_k, top_p)
#             elif len(img) == 2:
#                 res = ''
#                 s = ['satellite', 'street view']
#                 for i in range(2):
#                     system = f'You are analyzing {s[i]} image. ' + system
#                     r = self.chat(system, prompt, img[i], temp, top_k, top_p)
#                     res += r + '####################'
#                 return res
#     def chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
#         res = ollama.chat(
#             model='llama3.2-vision',
#             messages=[
#                 {
#                     'role': 'system',
#                     'content': system
#                 },
#                 {
#                     'role': 'user',
#                     'content': prompt,
#                     'images': [img]
#                 }
#             ],
#             options={
#                 "temperature":temp,
#                 "top_k":top_k,
#                 "top_p":top_p
#             }
#         )
#         return res['message']['content']

@st.cache_data
def convert_df(data):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return data.to_csv().encode("utf-8")

#--------------- User Interface & functions ----------------

parcels_, image_, output, tempr, top_k, top_p, system_info, prompt, sv_key, epsg_input = [None for i in range(10)]

###### sidebar ######
with st.sidebar:

    # parameters
    tempr = st.slider("temperature", 0.0, 1.0, 0.5)
    top_k = st.slider("top_k", 0.0, 1.0, 0.5)
    top_p = st.slider("top_p", 0.0, 1.0, 0.5)

    # prompt area
    system_info = st.text_area("Enter system context (optional):")
    prompt = st.text_area("Enter your prompt (required):")

######## tabs ########
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
            gdf = parcels_.drop(columns=['geometry'])
            st.dataframe(gdf)

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
    para_key, para_epsg = st.columns(2)
    with para_key:
        # text input for api key
        sv_key = st.text_input("api key (reqired)👇")
    with para_epsg:
        # enter epsg code
        epsg_input = st.number_input("epsg", value=None, placeholder="Type a epsg code...", step=1)
    
    # buttons for uploading files
    parcel_uploader_ = st.file_uploader("Upload parcel data_(required)", type=["zip"])
    if parcel_uploader_:
        st.write(f"You uploaded {parcel_uploader_.name}")
        if parcel_uploader_:
            parcels_ = loadSHP(parcel_uploader_)
        

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

if parcels_ is not None:
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

    with tab_parcel_streetview:
        sample_on = st.toggle("randonmly sample 2 rows for testing", True)
        btn1, btn2 = st.columns(2)
        with btn1:
            btn_send = st.button("process data", 
                                key="button_load_model_", 
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
                res = inputData.loopParcelChat(system=system_info, 
                                            prompt=prompt, 
                                            temp=tempr, 
                                            top_k=top_k, 
                                            top_p=top_p, 
                                            withSV=True,
                                            epsg=epsg_input,
                                            key=sv_key)
                # convert to dataframe
                output = pd.DataFrame(res)
        if output is not None:
            st.dataframe(output)
            csv = convert_df(output)
            # download the result as csv
            with btn2:
                st.download_button(label="Export result as CSV", data=csv, file_name="output.csv", mime="text/csv")
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
