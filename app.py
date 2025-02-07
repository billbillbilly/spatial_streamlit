from langchain_community.llms import Ollama 
import streamlit as st
from PIL import Image
import base64
import rasterio
import geopandas as gpd
from samgeo import tms_to_geotiff

st.header('🎈 R x Python Streamlit App for Spatial Data Analysis')


#--------------- Methods ---------------

class processData:
    def __init__(self, image=None, images=None):
        self.img = self.loadImg(image)
        self.imgs = images

    def loadImg(self, img):
        return base64.b64encode(img).decode("utf-8")

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
        tms_to_geotiff(output=map, bbox=bbox, zoom=19, source="Satellite", overwrite=True)
        return map
    
    def runLLMs(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        if prompt != None:
            llm = Ollama(model="llama3.2-vision")
            prompt = f"<image>{self.img}</image>\n{prompt}"
            return llm.invoke(prompt, stop=['<|eot_id|>'])
                
#--------------- User Interface & functions ----------------

# buttons for uploading files
parcel_uploader = st.file_uploader("Upload parcel data", type=["zip", "shp"])
if parcel_uploader:
    st.write(f"You uploaded {parcel_uploader.name}")
building_uploader = st.file_uploader("Upload building footprints data", type=["zip", "shp"])
if parcel_uploader:
    st.write(f"You uploaded {building_uploader.name}")
img_uploader = st.file_uploader("Upload image data", type=["png", "jpg", "jpeg", "tif"])
if img_uploader:
    image_bytes = img_uploader.read()
    st.write(f"You uploaded {img_uploader.name}")
    # image preview


# buttons for processing data
btn_load_data = st.button("load", key="button_load", help="click to load data", type='secondary', disabled=False)
if btn_load_data and image_bytes:
    inputData = processData(image=image_bytes)

# prompt area
system_info = st.text_area("Enter system context (optional):")
prompt = st.text_area("Enter your prompt:")

# buttons for sending prompts/running model
btn_send = st.button("send", key="button_send", help="click to send your prompt", type='secondary', disabled=False)
if btn_send:
    if inputData.img:
        inputData.runLLMs(prompt=prompt)