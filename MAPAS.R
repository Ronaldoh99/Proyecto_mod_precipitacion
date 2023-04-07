#Lectura de paquetes
library(sf)
library(ggplot2)
library(rnaturalearth)
library(rnaturalearthdata)
library(devtools)
#devtools::install_github("nebulae-co/colmaps", force = TRUE)
library(colmaps)
library(ggplot2)
library(sp)
#install.packages("devtools")
library("colmaps")
head(municipios@data)
library("ggplot2")
library(colmaps)
library(ggplot2)
library(sp)
library(colmaps)
library(ggplot2)
library(sp)
library(sf)
library(dplyr)
library(gpclib)
library(broom)
#Mapa de colombia por municipios
colmap(municipios) +
  ggtitle("Colombia - Fronteras Municipales")



mapameta <- departamentos[departamentos$id == 50,]
metamarca <- municipios[municipios@data$id_depto == 50,]

#Departamento del meta
colmap(map = metamarca, map_id = 'region', autocomplete = TRUE)



#MAPA de colombia
#por municipios
colmap(municipios) +
  ggtitle("Colombia - Fronteras Municipales")
#por departamentos
colmap(departamentos) +
  ggtitle("Colombia - Fronteras Municipales")

library(sf)
if (!requireNamespace("sf", quietly = TRUE)) {
  install.packages("sf")
}
# Crear una columna en el objeto departamentos que indique si el departamento es Meta o no
departamentos@data$is_meta <- ifelse(departamentos@data$id == 50, "Meta", "Otros departamentos")

# Convertir los datos de departamentos a formato sf
departamentos_sf <- st_as_sf(departamentos)

# Crear el gráfico base
mapa_ggplot <- ggplot() +
  geom_sf(data = departamentos_sf, aes(fill = is_meta), color = "black", size = 0.1) +
  scale_fill_manual(values = c("Otros departamentos" = "white", "Meta" = "dodgerblue"), name = "Departamentos", labels = c("Otros departamentos", "Meta")) +
  coord_sf()

# Agregar el título y ajustar los ejes para mostrar longitudes y latitudes
mapa_ggplot +
  ggtitle("Colombia - Fronteras Municipales") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  labs(x = "Longitud", y = "Latitud") +
  theme(axis.text.x = element_text(size = 8), axis.text.y = element_text(size = 8))


##MAPA de EL META

# Crear una función de escala de colores personalizada con tonos de azul aguamarina
escala_colores <- function(n) {
  colorRampPalette(c("deepskyblue4", "deepskyblue", "aquamarine"))(n)
}

# Agregar una columna de colores a los datos de municipios
metamarca@data$colores <- escala_colores(nrow(metamarca))

# Crear el gráfico
mapa_ggplot <- colmap(map = metamarca, map_id = 'region', autocomplete = TRUE)

# Ajustar la escala de colores en función de la columna creada previamente
mapa_ggplot <- mapa_ggplot + scale_fill_manual(values = metamarca@data$colores)

# Agregar el título y ajustar los ejes para mostrar longitudes y latitudes
mapa_ggplot +
  ggtitle("Mapa del departamento del Meta") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  labs(x = "Longitud", y = "Latitud") +
  theme(axis.text.x = element_text(size = 8), axis.text.y = element_text(size = 8))





# Crear una función de escala de colores personalizada con tonos de azul aguamarina
escala_colores <- function(n) {
  colorRampPalette(c("deepskyblue4", "deepskyblue", "aquamarine"))(n)
}

# Agregar una columna de colores a los datos de municipios
metamarca@data$colores <- escala_colores(nrow(metamarca))

# Crear una columna en el objeto metamarca que indique si el municipio es Villavicencio o no
metamarca@data$is_villavicencio <- ifelse(metamarca@data$id == 50001, "Villavicencio", "Otros municipios")

# Convertir los datos de metamarca a formato sf
metamarca_sf <- st_as_sf(metamarca)

# Crear el gráfico base
mapa_ggplot <- ggplot() +
  geom_sf(data = metamarca_sf, aes(fill = colores), color = "black", size = 0.1) +
  scale_fill_manual(values = metamarca@data$colores, name = "Municipios", labels = metamarca@data$nombre_mun) +
  coord_sf()

# Resaltar el municipio de Villavicencio y ajustar la leyenda
mapa_ggplot <- mapa_ggplot +
  geom_sf(data = metamarca_sf[metamarca_sf$is_villavicencio == "Villavicencio", ], aes(fill = is_villavicencio), color = "orange", size = 0.1) +
  scale_fill_manual(values = c("Villavicencio" = "blue", "Villavicencio" = "red"), name = "Municipios", labels = c("Villavicencio", "Villavicencio"), guide = "legend")

# Agregar el título y ajustar los ejes para mostrar longitudes y latitudes
mapa_ggplot +
  ggtitle("Mapa del departamento del Meta") +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  labs(x = "Longitud", y = "Latitud") +
  theme(axis.text.x = element_text(size = 8), axis.text.y = element_text(size = 8))




