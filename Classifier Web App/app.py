from flask import Flask, request, render_template, redirect, url_for, session
import os
import pandas as pd
import soundfile as sf
from maad import sound
from maad.util import power2dB, plot2d, overlay_rois, format_features
from maad.rois import create_mask, select_rois
from maad.features import shape_features, centroid_features
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'YSL'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

hopespotLinks = [
    {'Abrolhos Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A159'},
    {'Aeolian Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A284'},
    {'Agulhas Front': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A289'},
    {'Alboran Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A249'},
    {'Algoa Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A166'},
    {'Aliwal Shoal': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A167'},
    {'Andaman Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A168'},
    {'Argyll Coast and Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A158'},
    {'Ascension Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A169'},
    {'Atlantis Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A170'},
    {'Azores Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A300'},
    {'Bahamian Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A171'},
    {'Balearic Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A172'},
    {'Bering Sea Deep Canyons': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A173'},
    {'Biological Marine Corridor of Osa': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A322'},
    {'Blue Cavern State Marine Conservation Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A352'},
    {'Blue Shark Central': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A174'},
    {'Bocas del Toro Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A266'},
    {'Bunaken Marine Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A271'},
    {'Byron Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A343'},
    {'Cagarras Islands and Surrounding Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A287'},
    {'California Seamounts': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A175'},
    {'Canyon of Caprera': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A350'},
    {'Cape Whale Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A176'},
    {'Capurgana and Cabo Tiburon': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A254'},
    {'Cashes Ledge': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A177'},
    {'Central Arctic Ocean': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A179'},
    {'Chagos Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A180'},
    {'Charlie-Gibbs Fracture Zone': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A181'},
    {'Chichiriviche de la Costa': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A182'},
    {'Chilean Fjords and Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A183'},
    {'Choroni and Chuao': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A184'},
    {'Chumash Heritage National Marine Sanctuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A351'},
    {'Coastal Southeast Florida': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A185'},
    {'Coastal Waters of the Black River District': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A162'},
    {'Cocos Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A280'},
    {'Cocos-Galápagos Swimway': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A267'},
    {'Coiba and Cordillera de Coiba': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A281'},
    {'Conflict Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A269'},
    {'Coral Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A292'},
    {'Coral Seamount': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A187'},
    {'Coral Triangle': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A188'},
    {'Costa Rica Thermal Dome': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A178'},
    {'Datan Algal Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A255'},
    {'East Antarctic': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A190'},
    {'East Portland Fish Sanctuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A191'},
    {'Eastern Tropical Pacific Seascape': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A192'},
    {'Egg Island, Bahamas': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A193'},
    {'Emperor Seamount Chain': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A194'},
    {'Exmouth Gulf and Ningaloo Coast World Heritage Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A164'},
    {'False Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A195'},
    {'Farm Pond': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A196'},
    {'Fish Rock': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A309'},
    {'Florida Gulf Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A160'},
    {'French Overseas Territories (Wallis and Futuna)': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A197'},
    {'Gakkel Ridge': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A198'},
    {'Galápagos Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A279'},
    {'George Town': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A200'},
    {'Georgia Continental Shelf and Blake Plateau': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A263'},
    {'Ghizilagaj Reserve National Park and Marine Protected Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A260'},
    {'Gold Coast Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A278'},
    {'Golfo Dulce': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A256'},
    {'Gotland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A277'},
    {'Grand Recif de Toliara': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A201'},
    {'Great Barrier Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A288'},
    {'Great Lakes': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A345'},
    {'Greater Farallones': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A205'},
    {'Greater Skellig Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A342'},
    {'Guanahacabibes National Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A250'},
    {'Gulf of California': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A202'},
    {'Gulf of Guinea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A203'},
    {'Gulf of Mexico Deep Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A204'},
    {'Hatteras': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A206'},
    {'Hecate Strait and Queen Charlotte Sound Glass Sponge Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A207'},
    {'Henoko-Ōura Coastal Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A262'},
    {'Hong Kong South': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A334'},
    {'Houtman Abrolhos': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A208'},
    {'Humboldt Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A247'},
    {'Inhambane Seascape': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A326'},
    {'Jaeren Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A248'},
    {'Jangamo Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A330'},
    {'Jardines de la Reina': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A199'},
    {'Kahalu’u Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A328'},
    {'Kangaroo Island North Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A283'},
    {'Kep Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A257'},
    {'Kermadec Trench': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A209'},
    {'Kimbe Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A261'},
    {'Knysna Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A210'},
    {'Kosterfjorden Yttre Hvaler': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A211'},
    {'Laamu Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A324'},
    {'Lakshadweep Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A212'},
    {'Lesvos': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A290'},
    {'Little Cayman': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A275'},
    {'Long Island Marine Management Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A252'},
    {'Lord Howe Rise': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A291'},
    {'Mako Shark Metropolis': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A213'},
    {'Maldive Atolls': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A214'},
    {'Malpelo Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A215'},
    {'Maputaland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A355'},
    {'Mesoamerican Reefs': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A217'},
    {'Misool Marine Reserve': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A219'},
    {'Mohéli': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A282'},
    {'Monterey Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A220'},
    {'Moreton Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A221'},
    {'Myeik Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A216'},
    {'New York-New Jersey Harbor Estuary': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A341'},
    {'New Zealand Coastal Waters': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A270'},
    {'Northeast Iceland': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A344'},
    {'Northwest Passage': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A222'},
    {'Nusa Penida Marine Protected Area': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A268'},
    {'Ocean Cay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A337'},
    {'Olowalu Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A245'},
    {'Ombai-Wetar Strait': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A276'},
    {'Outer Islands of Seychelles': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A223'},
    {'Pacific Subtropical Convergence Zone': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A224'},
    {'Palau': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A218'},
    {'Palmahim Slide': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A336'},
    {'Palmyra Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A273'},
    {'Pangatalan Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A165'},
    {'Patagonian Shelf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A225'},
    {'Pearl Islands Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A346'},
    {'Plettenberg Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A226'},
    {'Prince William Sound': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A348'},
    {'Quirimbas Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A227'},
    {'Revillagigedo Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A246'},
    {'Ross Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A228'},
    {'Saba and the Saba Bank': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A338'},
    {'Saint Barthélemy': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A331'},
    {'Saint Vincent and the Grenadines': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A229'},
    {'Salas y Gomez and Nazca Ridges': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A230'},
    {'Salisbury Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A163'},
    {'Salish Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A285'},
    {'San Francisco Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A253'},
    {'Sargasso Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A319'},
    {'Scott Islands': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A232'},
    {'Shinnecock Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A333'},
    {'South San Jorge Gulf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A340'},
    {'Southeast Shoal of the Grand Banks': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A233'},
    {'St Helena Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A347'},
    {'Subantarctic Islands and Surrounding Seas': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A235'},
    {'Svalbard Archipelago': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A234'},
    {'Sydney Coast': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A258'},
    {'Tasman Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A236'},
    {'Tavarua Island': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A274'},
    {'Tenerife-La Gomera': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A265'},
    {'Tetiaroa Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A259'},
    {'The Great Fringing Reef of the Red Sea': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A339'},
    {'The Great Southern Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A264'},
    {'Tribuga Gulf': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A161'},
    {'Tropical Pacific Sea of Peru': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A238'},
    {'Tubbataha Reefs Natural Park': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A329'},
    {'Varadero’s Coral Reef': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A272'},
    {'Verde Island Passage': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A349'},
    {'Walter’s Shoal': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A240'},
    {'Western Pacific Donut Hole 1': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A241'},
    {'Western Pacific Donut Hole 2': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A242'},
    {'Western Pacific Donut Hole 3': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A243'},
    {'Western Pacific Donut Hole 4': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A244'},
    {'Whale and Dolphin Sanctuary of Uruguay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A251'},
    {'White Shark Cafe': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A237'},
    {'Wider Vatika Bay': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A239'},
    {'Wotho Atoll': 'https://experience.arcgis.com/experience/aa70b7e4739c4b33a0d7e21c85b88f8e#data_s=id%3AdataSource_5-17d9073e68b-layer-27-17d9073e6c7-layer-28%3A312'}
]

def process_audio(file_path):
    # Load the audio file
    s, fs = sound.load(file_path)
    s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and smoothing
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.25)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Calculate shape and centroid features
    df_shape, params = shape_features(Sxx_db, resolution='low', rois=df_rois)
    df_centroid = centroid_features(Sxx_db, df_rois)

    # Get median frequency and normalize
    median_freq = fn[np.round(df_centroid.centroid_y).astype(int)]
    df_centroid['centroid_freq'] = median_freq / fn[-1]

    # t-SNE for dimensionality reduction
    X = df_shape.loc[:, df_shape.columns.str.startswith('shp')]
    X = X.join(df_centroid.centroid_freq)  # add column and normalize values

    tsne = TSNE(n_components=2, perplexity=12, init='pca', verbose=True)
    Y = tsne.fit_transform(X)

    # Clustering using DBSCAN
    cluster = DBSCAN(eps=5, min_samples=4).fit(Y)

    # Overlay bounding box on the original spectrogram
    df_rois['label'] = cluster.labels_.astype(str)
    overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':60, 'extent':ext})

    # Filter ROIs for those with centroid frequency below 1000Hz
    low_freq_rois = df_rois[df_centroid['centroid_freq'] * fn[-1] < 1000]

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    audio_clips = []
    for i, row in low_freq_timestamps.iterrows():
        # Add half a second (0.5 seconds) to the beginning and end of the clip
        start_sample = int(max(0, (row['begin'] - 0.5) * fs))
        end_sample = int(min(len(s), (row['end'] + 0.5) * fs))
        audio_clip = s[start_sample:end_sample]
        clip_filename = f'clip_{i}.wav'
        clip_path = os.path.join(app.config['UPLOAD_FOLDER'], clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append(clip_filename)

    low_freq_timestamps['audio_clip'] = audio_clips

    return low_freq_timestamps

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    location = request.form.get('location')
    
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        all_timestamps = []
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        timestamps = process_audio(file_path)
        all_timestamps.append(timestamps)

        
        session['all_timestamps'] = [df.to_dict(orient='records') for df in all_timestamps]
        session['titles'] = timestamps.columns.values.tolist()
        session['location'] = location
        
        
        return redirect(url_for('result'))

@app.route('/result')
def result():
    all_timestamps = [pd.DataFrame(data) for data in session.get('all_timestamps', [])]
    titles = session.get('titles', [])
    location = session.get('location', '')
    
    locationLink = searchDict(hopespotLinks, location)
    
    return render_template('result.html', tables=all_timestamps, titles=titles, location=location, locationLink=locationLink)


from flask import send_file
import zipfile
import io

@app.route('/download_all')
def download_all():
    # Create a BytesIO buffer to write the zip file
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for clip in session.get('all_timestamps', [])[0]:
            clip_path = os.path.join(app.config['UPLOAD_FOLDER'], clip['audio_clip'])
            zf.write(clip_path, arcname=os.path.basename(clip_path))
    
    memory_file.seek(0)
    return send_file(memory_file, download_name='all_clips.zip', as_attachment=True)



def searchDict(dict, search):
    for item in dict:
        if search in item:
            return item[search]
    return None

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
