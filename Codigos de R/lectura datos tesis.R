#lectura de datos 
load("C:/Users/ronal/Desktop/SEMETRE 9/TRABAJO DE GRADO 1/Proyecto_mod_precipitacion/Datos/wth_data_caf_ideam.RData") # nolint # nolint: line_length_linter.

#Lectura de paquetes
instalar_si_no_existe <- function(nombre_paquete) {
  if (!requireNamespace(nombre_paquete, quietly = TRUE)) {
    install.packages(nombre_paquete)
  }
  library(nombre_paquete, character.only = TRUE)
}

# Lista de paquetes necesarios
paquetes <- c("sf", "ggplot2", "rnaturalearth", "rnaturalearthdata", "devtools", 
              "colmaps", "sp", "dplyr", "gpclib", "broom","leaflet","tidyverse",
              "purrr","ggspatial","RColorBrewer","utils")

# Instalar e importar paquetes
lapply(paquetes, instalar_si_no_existe)

# Instalar colmaps desde GitHub
if (!"colmaps" %in% rownames(installed.packages())) {
  devtools::install_github("nebulae-co/colmaps", force = TRUE)
}

# Importar colmaps
library(colmaps)

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
leaflet() %>% addTiles() %>% addCircleMarkers(lng = ws_selected$lon,lat = ws_selected$lat)



#Estaciones de Meta Villa Vicencio 
library(dplyr)
datosmeta = ws_selected %>% filter(ws_selected$Departamento=="META" & ws_selected$var=="prec")
str(datosmeta)

#mapa interactivo del meta 
leaflet() %>% addTiles() %>% addCircleMarkers(lng = datosmeta$lon,lat = datosmeta$lat)


#Tabla de datos por estaciones para presentar
datostable <- data.frame(datosmeta$id,datosmeta$Nombre,datosmeta$lat,datosmeta$lon,datosmeta$na_percent)
head(datostable,84)

# Exportar el data.frame a un archivo de Excel
library(writexl)
#write_xlsx(datostable, "datostable.xlsx")


#Graficos de series para cada estacion

#Estacion 1
datosmeta[[8]][[1]]#Nombre de la estacion
datosmeta[[3]][[1]]$Date
datosmeta[[3]][[1]]$prec
library(ggplot2)

# Crear el gráfico de línea
ggplot(data = datosmeta[[3]][[1]], aes(x = datosmeta[[3]][[1]]$Date, y = datosmeta[[3]][[1]]$prec)) +
  geom_line() +
  ggtitle("Serie temporal de Precipitaciom - Estacion MARIPOSA LA [32010010]") +
  xlab("Tiempo") +
  ylab("Precipitación") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))


#Estacion 1
datosmeta[[8]][[2]]#Nombre de la estacion
datosmeta[[3]][[2]]$Date



# Crear el gráfico de línea
ggplot(data = datosmeta[[3]][[2]], aes(x = datosmeta[[3]][[2]]$Date, y = datosmeta[[3]][[2]]$prec)) +
  geom_line() +
  ggtitle("Serie temporal de Precipitaciom - Estacion BOCAS DEL DUDA") +
  xlab("Tiempo") +
  ylab("Precipitación") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))







library(ggplot2)
library(dplyr)

# Combinar todos los data.frames en un solo data.frame y agregar una columna con el índice
datos_combinados <- data.frame()

for (i in 1:12) {
  temp_df <- datosmeta[[3]][[i]]
  temp_df$indice <- i
  datos_combinados <- rbind(datos_combinados, temp_df)
}

# Crear el gráfico de línea con múltiples paneles
ggplot(data = datos_combinados, aes(x = Date, y = prec)) +
  geom_line() +
  facet_wrap(~ indice, ncol = 4, scales = "free_x") +
  ggtitle("Serie temporal de Precipitaciones") +
  xlab("Tiempo") +
  ylab("Precipitación") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        strip.text.x = element_text(size = 10, face = "bold"))



# Combinar todos los data.frames en un solo data.frame y agregar una columna con el índice y el nombre
datos_combinados <- data.frame()

for (i in 1:12) {
  temp_df <- datosmeta[[3]][[i]]
  temp_df$indice <- i
  temp_df$nombre <- datosmeta[[8]][[i]]
  datos_combinados <- rbind(datos_combinados, temp_df)
}

# Crear el gráfico de línea con múltiples paneles y nombres personalizados
ggplot(data = datos_combinados, aes(x = Date, y = prec)) +
  geom_line() +
  facet_wrap(~ nombre, ncol = 4, scales = "free_x") +
  ggtitle("Serie temporal de Precipitaciones") +
  xlab("Tiempo") +
  ylab("Precipitación") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        strip.text.x = element_text(size = 10, face = "bold"))


############# GRAFICO CIRCULO CORRELACION #############################
library(ggcorrplot)
library(tidyverse)
library(purrr)


#Transformacion de los datos para sacar los dataframes y tener las variables fecha y prec
datosmeta

datos_combinados <- datosmeta %>%
  mutate(combined = map2(id, data, ~ mutate(.y, ID = .x))) %>%
  select(combined) %>%
  unnest(cols = c(combined))

print(datos_combinados)         #observar los datos
is.data.frame(datos_combinados) #confirmar que es data.frame
str(datos_combinados)
#Exportacion de datos en formato para modelo KNN que se ejecutara en python

library(utils)
#write.csv2(datos_combinados, "Datos/datos_combinados_para_modelo.csv", row.names = FALSE)
#write.table(datos_combinados, "Datos/datos_combinados_para_modelo.txt", sep = ";", row.names = FALSE, col.names = TRUE, quote = FALSE)
#datosprueba1<-read.csv2("datos_combinados_para_modelo.csv")
#str(datosprueba1)
#datosprueba2<-read.csv("Datos/datos_combinados_para_modelo.txt",sep = ";")
#str(datosprueba2)


#Tranformar para que se guarde las columnas como nombre del data frame

# Asumiendo que 'datos_combinados' es el data.frame que obtuvimos en el paso anterior

datos_wider <- datos_combinados %>%
  tidyr::pivot_wider(names_from = ID, values_from = prec)

print(datos_wider)



###MATRIZ DE CORRELACION HECHO POR COMPLETAR LAS CORRELACIONES POR DONDE SE ENCUENTRA UNA INTERCEPCION
AQ.cor = cor(datos_wider[,-1],method="pearson",use = "pairwise.complete.obs")
print(AQ.cor)

#GRAFICO DE MATRIZ DE CORRELACION 
windows(height=10,width=15)
corrplot::corrplot(AQ.cor, method = "ellipse",addCoef.col = "black",type="upper")

####################################################################################
##################               mapa de estaciones    ############################
####################################################################################

#combinacion de datos 
datos_combinadoscalor <- datosmeta %>%
  mutate(combined = pmap(list(id, lat, lon, Nombre, data), function(id, lat, lon, Nombre ,data) {
    mutate(data, ID = id, LAT = lat, LON = lon, NOMBRE = Nombre)
  })) %>%
  select(combined) %>%
  unnest(cols = c(combined))

print(datos_combinadoscalor)

datosaexportar<- datos_combinadoscalor[,c(1:5)]
datosaexportar
#datos mas completos
write.table(datosaexportar, "Datos/datos_combinados_para_modelo_con_coordenadas.txt", sep = ";", row.names = FALSE, col.names = TRUE, quote = FALSE)


library(ggplot2)
library(ggspatial)
library(RColorBrewer)

# Crear un conjunto de datos de ejemplo con longitud, latitud y precipitación para Villavicencio
example_data <- data.frame(
  longitude = c(-73.635, -73.605, -73.623, -73.656),
  latitude = c(4.151, 4.111, 4.155, 4.167),
  precipitation = c(1200, 650, 800, 1200)
)

# Cargar un mapa base de OpenStreetMap
map_base <- ggmap::get_map(location = c(-75, 1, -71, 5), source = "stamen", maptype = "terrain")

# Crear el mapa de con los puntos de las estciones con ggplot2
ggmap::ggmap(map_base) +
  geom_point(data = datos_combinadoscalor, aes(x = datos_combinadoscalor$LON, y = datos_combinadoscalor$LAT, color = datos_combinadoscalor$prec, size = datos_combinadoscalor$prec), alpha = 0.6) +
  scale_color_gradientn(colors = brewer.pal(9, "YlGnBu"), name = "Precipitación") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = 12), legend.text = element_text(size = 10))





