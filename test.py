import folium
from folium import plugins
import pandas as pd
import numpy as np


fpath = '/content/drive/My Drive/RMDSLAB_test.csv'
output_path = '/content/'



def scaler(lst):
    new_lst = []
    vmax = max(lst)
    vmin = min(lst)
    for x in lst:
        new_lst.append((x-vmin)/(vmax-vmin))
    return new_lst

def plot(heatmap_data, idx, init_center):

    heatmap = folium.Map(location = init_center, tiles = 'CartoDB Positron', 
                        zoom_start = 3,
                        control_scale = True)
    plugins.HeatMapWithTime(data = heatmap_data,
                            auto_play=True,
                            max_opacity=0.8, 
                            index = idx, 
                            radius = 25,
                            name = 'Target').add_to(heatmap)
    folium.raster_layers.TileLayer('Open Street Map').add_to(heatmap)
    folium.raster_layers.TileLayer('Stamen Terrain').add_to(heatmap)
    folium.raster_layers.TileLayer('Stamen Toner').add_to(heatmap)
    heatmap.add_child(folium.LatLngPopup())
    measure_control = plugins.MeasureControl(primary_length_unit = 'kilometers')
    heatmap.add_child(measure_control)
    heatmap.add_child(folium.LayerControl())
    return heatmap

def main():
    try: 
        assert folium.__version__ == '0.11.0'
    except:
        print('folium version may not be high enough to display time variant map correctly!')

    df = pd.read_csv(fpath)
    time_idx = df.timestamp.apply(lambda x: f'{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]} {str(x)[8:]}H').values.tolist()
    df['scaled_dvmax_score'] = scaler(df.predicted_dvmax_score.values)
    heatmap_dvmax = []
    for i in df.index.values:
        heatmap_dvmax.append([df[['latitude','longitude','scaled_dvmax_score']].loc[i].values.tolist()])
    df[' predicted_RI_class'] = df[' predicted_RI_class'].map({0:0.001, 1:1})
    heatmap_RI = []
    for i in df.index.values:
        heatmap_RI.append([df[['latitude','longitude', ' predicted_RI_class']].loc[i].values.tolist()])

    dvmax_heatmap = plot(heatmap_dvmax, time_idx, [df.latitude.mean(), df.longitude.mean()] )
    dvmax_RI = plot(heatmap_RI, time_idx, [df.latitude.mean(), df.longitude.mean()] )
    dvmax_heatmap.save(output_path + 'dvmax_heatmap.html')
    dvmax_RI.save(output_path + 'dvmax_RI.html')
main()
