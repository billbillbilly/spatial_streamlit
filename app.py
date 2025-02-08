
import streamlit as st
from PIL import Image
import ollama

import os
import zipfile
import tempfile
import leafmap.foliumap as leafmap
import rasterio
import geopandas as gpd
# from samgeo import tms_to_geotiff

st.header('🎈 Streamlit App for Spatial Data Analysis')

#--------------- Methods ---------------

class processData:
    def __init__(self, image=None, images=None):
        self.img = image
        self.imgs = images

    def loadParcelsAndMap(self, parcels=None, buildings=None, map=None, bbox=None):
        if parcels != None:
            # import shp data
            p = gpd.read_file(parcels)
            if buildings != None:
                b = gpd.read_file(buildings)
                # Ensure both layers have the same CRS
                if p.crs != b.crs:
                    b = b.to_crs(p.crs)
                # select parcels with buildings
                self.filtered_parcels = gpd.overlay(p, b, how="intersection")
                # Save the filtered parcels
                self.filtered_parcels.to_file("filtered_polygons.shp")
            else:
                self.filtered_parcels = parcels
            # import map
            if map != None:
                m = rasterio.open(map)
            if bbox != None:
                m = rasterio.open(self.getMap(bbox))
    
    def getMap(self, bbox):
        map = "map.tif"
        # tms_to_geotiff(output=map, bbox=bbox, zoom=19, source="Satellite", overwrite=True)
        return map
    
    def oneImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        return self.LLM_chat(system=system, prompt=prompt, img=self.img, 
                             temp=temp, top_k=top_k, top_p=top_p)
    
    def LLM_chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
        if prompt != None and img != None:
            res = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img]
                }]
            )
            return res['message']['content']

def loadSHP (uploaded_file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the uploaded ZIP file in the temporary directory
        zip_path = os.path.join(tmp_dir, "uploaded_shapefile.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find the .shp file in the extracted files
        shp_file = None
        for root, dirs, files in os.walk(tmp_dir):
            # Skip __MACOSX and hidden files
            if "__MACOSX" in root:
                continue

            for file in files:
                if file.startswith("._"):  # Skip macOS metadata files
                    continue

                if file.endswith(".shp"):
                    shp_file = os.path.join(root, file)
                    break

        # Read the Shapefile using GeoPandas
        if shp_file:
            try: 
                displaySHP(shp_file)
            except Exception as e:
                st.error(f"Error reading Shapefile: {e}")

def displaySHP(data):
    # Display map in Streamlit
    m = leafmap.Map()
    m.add_shp(data, layer_name="parcels")
    m.to_streamlit(height=400)

#--------------- User Interface & functions ----------------

image_ = None
parcels_ = None
uploadingCol, displayCol = st.columns(2)
# buttons for uploading files
with uploadingCol:
    parcel_uploader = st.file_uploader("Upload parcel data [.zip]", 
                                       type=["zip"], 
                                       label_visibility="hidden",
                                       help="Upload a ZIP file containing .shp, .shx, .dbf, and other necessary files")
    if parcel_uploader:
        if parcel_uploader is not None:
            #parcels_ = loadSHP(parcel_uploader)
            st.write(f"You uploaded {parcel_uploader.name}")
    building_uploader = st.file_uploader("Upload building footprints data", type=["zip"])
    if building_uploader:
        st.write(f"You uploaded {building_uploader.name}")
    img_uploader = st.file_uploader("Upload image data", type=["png", "jpg", "jpeg", "tif"])
    if img_uploader:
        image_ = img_uploader.read()
        st.write(f"You uploaded {img_uploader.name}")
with displayCol:
    displayNames = ["show image", "show parcels", "show buildings"]
    display_ = st.radio("dispaly data", displayNames)
    if display_ == "show image":
        if image_:
            st.image(image_, caption="input image")
    elif display_ == "show parcels":
        if parcel_uploader:
            loadSHP(parcel_uploader)
    elif display_ == "show buildings":
        st.write(f"You uploaded buildings")

# interactive map
mapView = st.toggle("get image on map")
bbox_ = None
if mapView:
    m = leafmap.Map(minimap_control=True)
    m.add_basemap("SATELLITE")
    m.to_streamlit(height=500)
    # buttons for clip map
    clip_map = st.button("clip map with the polygon", 
                         key="button_clip", 
                         help="click to clip map with polygon", 
                         type='secondary', 
                         disabled=False)
    if clip_map:
        bbox_ = m.user_roi_bounds()
        st.write(f'{bbox_}')

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
    prompt = st.text_area("Enter your prompt:")

# buttons for sending prompts/running model
btn_send = st.button("send", key="button_send", help="click to send your prompt", type='secondary', disabled=False)
if btn_send:
    inputData = processData(image=image_)
    if inputData != None:
        res = inputData.oneImgChat(prompt=prompt, temp=tempr, top_k=top_k, top_p=top_p)
        st.write(res)