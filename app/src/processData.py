import streamlit as st
import ollama
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from samgeo import tms_to_geotiff
import requests
import numpy as np
from pyproj import Transformer
from pyproj import CRS
import math
from PIL import Image
import sys
import cv2
from .pano2pers import Equirectangular

class processData:
    def __init__(self, image=None, images=None, parcels=None):
        self.img = image
        self.imgs = images
        self.parcels = parcels
    
    def getSV(self, centroid, epsg, key):
        bbox = self.projection(centroid, epsg)
        url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_1024_url,geometry&bbox={bbox}&is_pano=true"
        # response = None
        # while not response or 'data' not in response:
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
        # reframe image
        svi = Equirectangular(img_url=img_url)
        pers, sv = svi.GetPerspective(80, relative_heading, 10, 300, 400, 128)
        cv2.imwrite('cropped_sv.png', pers)
        # # Download Image
        # sv_data = requests.get(img_url).content
        # with open("sv.jpg", "wb") as handler:
        #     handler.write(sv_data)
        # # Load 360 Image
        # image = Image.open("sv.jpg")
        # width, height = image.size 
        # # Convert relative heading to pixel offset
        # crop_width = width // 4  # Approximate a 90-degree field of view
        # center_x = int((relative_heading / 360) * width)
        # # Crop Image
        # left = max(0, center_x - crop_width // 2)
        # right = min(width, center_x + crop_width // 2)
        # cropped_image = image.crop((left, 0, right, height))
        # cropped_image.save("cropped_sv.jpg", format="JPEG")
        print('street view downloaded')
        return sv
    
    def projection(self, centroid, epsg):
        x, y = self.degree2dis(centroid, epsg)
        # Get unit name (meters, degrees, etc.)
        crs = CRS.from_epsg(epsg)
        unit_name = crs.axis_info[0].unit_name
        # set search distance to 25 meters
        r = 50
        if unit_name == 'foot':
            r = 164.042
        elif unit_name == 'degree':
            print("Error: epsg must be projected system.")
            sys.exit(1)
        # set bbox
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
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        delta_lon = lon2 - lon1

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360  # Normalize to 0-360
    
    def oneImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        return self.LLM_chat(system=system, prompt=prompt, img=[self.img], 
                             temp=temp, top_k=top_k, top_p=top_p)
    
    def loopImgChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None):
        for img in self.imgs:
            self.LLM_chat(system=system, prompt=prompt, img=[img], 
                          temp=temp, top_k=top_k, top_p=top_p)
            
    def loopParcelChat(self, system=None, prompt=None, temp=None, top_k=None, top_p=None, withSV=False, epsg=None, key=None):
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
            centroid = polygon.centroid
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

            # add images
            input_imgs = [clipped_image]

            if withSV == True and epsg != None and key != None:
                input_imgs += [self.getSV(centroid, epsg, key)]

            res = self.LLM_chat(system=system, 
                                prompt=prompt, 
                                img=input_imgs, 
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
            if len(img) == 1:
                return self.chat(system, prompt, img[0], temp, top_k, top_p)
            elif len(img) == 2:
                res = ''
                messages = []
                for i in range(2):
                    system = f'You are analyzing both satellite and street view images. For street view, you should just foucus on the building and yard in the middle. {system}' 
                    r, messages = self.streaming_chat(messages, system, prompt, img[i], temp, top_k, top_p)
                    res += r + '####################'
                return res
    def streaming_chat(self, messages=[], system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
        if system and len(messages) == 0:
            messages.append({
                'role': 'system',
                'content': system
            })
        message = [
            {
                'role': 'user',
                'content': prompt,
                'images': [img]
            }
        ]
        messages += message
        stream = ollama.chat(
            model='llama3.2-vision',
            messages=messages,
            options={
                "temperature":temp,
                "top_k":top_k,
                "top_p":top_p
            },
            stream=True
        )
        # Collect streaming assistant response
        assistant_content = ""
        for chunk in stream:
            content = chunk['message']['content']
            assistant_content += content
        # Add assistant message to chat history
        assistant_message = {
            'role': 'assistant',
            'content': assistant_content
        }
        messages.append(assistant_message)
        
        return assistant_content, messages

    def chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
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