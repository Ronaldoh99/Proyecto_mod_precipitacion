## este script es para leer los datos 
load("~/semestre 9/trabajo de grado/Proyecto_mod_precipitacion/Datos/wth_data_caf_ideam.RData")

load("C:/Users/ronal/Desktop/SEMETRE 9/TRABAJO DE GRADO 1/Proyecto_mod_precipitacion/Datos/wth_data_caf_ideam.RData")

##bases candidatas

#Candidata 1
View(ws_selected[[3]][[301]])

#cantidad de Años:
ws_selected[301,6]

#rango de años:
#Desde
ws_selected[301,4]
#Hasta
ws_selected[301,5]

#cantidad de NAs:
ws_selected[301,7]

#Origen de los datos o Region 
ws_selected[301,10]
ws_selected[301,11]

#longitud y latitud
ws_selected[301,12]
ws_selected[301,13]


#candidadata 2:
View(ws_selected[[3]][[569]])


#cantidad de Años:
ws_selected[569,6]

#rango de años:
#Desde
ws_selected[569,4]
#Hasta
ws_selected[569,5]

#cantidad de NAs:
ws_selected[569,7]

#Origen de los datos o Region 
ws_selected[569,10]
ws_selected[569,11]

#logitud y alatitud
ws_selected[569,12]
ws_selected[569,13]

#Graficos de candidatos 
library(leaflet)
leaflet() %>% addTiles() %>% addCircleMarkers(lng = c(-73.61758,-73.625),lat = c(4.161919,4.137389))


