
import streamlit as st
from PIL import Image
import ollama

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
                
#--------------- User Interface & functions ----------------

image_ = None
uploadingCol, displayCol = st.columns(2)
# buttons for uploading files
with uploadingCol:
    parcel_uploader = st.file_uploader("Upload parcel data", type=["zip"])
    if parcel_uploader:
        st.write(f"You uploaded {parcel_uploader.name}")
    building_uploader = st.file_uploader("Upload building footprints data", type=["zip"])
    if parcel_uploader:
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
        st.write(f"You uploaded pacels")
    elif display_ == "show buildings":
        st.write(f"You uploaded buildings")

# interactive map
mapView = st.toggle("get image on map")
if mapView:
    st.write('res')

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