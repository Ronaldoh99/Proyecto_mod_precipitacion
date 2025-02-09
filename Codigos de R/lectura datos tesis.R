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


datostable2 <- data.frame(datosmeta$id,datosmeta$Nombre,datosmeta$lat,datosmeta$lon,datosmeta$na_percent,datosmeta$idate,datosmeta$fdate,datosmeta$years)
head(datostable2,84)

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
# Combinar todos los data.frames en un solo data.frame y agregar una columna con el índice y el nombre
datosmeta<-datosmeta[-lista_a_eliminar,]
datos_combinados <- data.frame()


for (i in 38:56) {
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





#Transformacion de los datos para sacar los dataframes y tener las variables fecha y prec
datosmeta

datos_combinados <- datosmeta %>%
  mutate(combined = map2(id, data, ~ mutate(.y, ID = .x))) %>%
  select(combined) %>%
  unnest(cols = c(combined))

print(datos_combinados)         #observar los datos
is.data.frame(datos_combinados) #confirmar que es data.frame
str(datos_combinados)


#seleccion de estaciones para establece periodo de investigacion la cual es entre 1980 hasta el 2016 
#listado de numero de la fila para eliminar de la tabla datosmeta
lista_a_eliminar<-c(2,4,10,19,26,31,34,38,44,51,52,53,57,59,60,64,69,75,76,80,82,72,71,61,35,36,55)
length(lista_a_eliminar)#total de estaciones eliminadas 27

Datos_seleccionados <-datosmeta[-lista_a_eliminar,] %>%
  mutate(combined = map2(id, data, ~ mutate(.y, ID = .x))) %>%
  select(combined) %>%
  unnest(cols = c(combined))

print(Datos_seleccionados)         #observar los datos
is.data.frame(Datos_seleccionados) #confirmar que es data.frame
str(Datos_seleccionados)

#Exportacion de datos en formato para modelo KNN que se ejecutara en python

library(utils)
#write.csv2(datos_combinados, "Datos/datos_combinados_para_modelo.csv", row.names = FALSE)
#write.table(datos_combinados, "Datos/datos_combinados_para_modelo.txt", sep = ";", row.names = FALSE, col.names = TRUE, quote = FALSE)
#datosprueba1<-read.csv2("datos_combinados_para_modelo.csv")
#str(datosprueba1)
#datosprueba2<-read.csv("Datos/datos_combinados_para_modelo.txt",sep = ";")
#str(datosprueba2)

#exportar datos con estaciones filtradas 57 en total antes 84
#seleccion de estaciones para establece periodo de investigacion la cual es entre 1980 hasta el 2016 
#listado de numero de la fila para eliminar de la tabla datosmeta
lista_a_eliminar<-c(2,4,10,19,26,31,34,38,44,51,52,53,57,59,60,64,69,75,76,80,82,72,71,61,35,36,55)
length(lista_a_eliminar)#total de estaciones eliminadas 27

Datos_seleccionados <- datosmeta[-lista_a_eliminar,] %>%
  mutate(combined = pmap(list(id, lat, lon, Nombre, data), function(id, lat, lon, Nombre ,data) {
    mutate(data, ID = id, LAT = lat, LON = lon, NOMBRE = Nombre)
  })) %>%
  select(combined) %>%
  unnest(cols = c(combined))

print(Datos_seleccionados)

datosaexportar_seleccionados<- Datos_seleccionados[,c(1:5)]
datosaexportar_seleccionados
#datos mas completos

write.csv2(datosaexportar_seleccionados, "Datos/datos_seleccionados_para_modelo_coordenadas.csv", sep = ";", row.names = FALSE, col.names = TRUE, quote = FALSE)



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



#Grafico de Histograma y densidad del comportamiento de los resultados del RMSE



#Lectura de datos en formato para modelo
datos_precipitacion <-read.csv("Datos/datos_seleccionados_para_modelo_coordenadas.txt",na.strings = c("N/A", " ", "NA"),sep = ";")


library(readxl)
library(ggplot2)

metricas_por_id <- read_excel("F:/TESIS/Proyecto_mod_precipitacion/metricas_por_id.xlsx", 
                              range = "A1:H52")

str(metricas_por_id)

names(metricas_por_id)

# Datos
set.seed(5)
x <- rnorm(1000)
df <- data.frame(x)



df <- metricas_por_id
x <- metricas_por_id$RMSE
y <- metricas_por_id$MAE
z <- metricas_por_id$MAPE
w <- metricas_por_id$R2

# Histograma con densidad
ggplot(df, aes(x = x)) + 
  geom_histogram(aes(y = ..density..),
                 colour = 1, fill = "#00CDCD") +
  geom_density(lwd = 1, colour = 4,
               fill = 4, alpha = 0.25) 



# Asegúrate de tener cargado el paquete ggplot2
library(ggplot2)

library(gridExtra)


# Datos de ejemplo
df <- metricas_por_id

# Creamos una función para generar el histograma con densidad para una métrica específica
plot_histogram <- function(data, metric, color_fill, color_density, title, Xlab_) {
  ggplot(data, aes_string(x = metric)) + 
    geom_histogram(aes(y = ..density..), color = "black", fill = color_fill, bins = 30) +
    geom_density(color = color_density, fill = color_density, alpha = 0.25) +
    labs(title = title, x = Xlab_ , y = "Densidad") +
    theme_minimal()+
    theme(plot.title = element_text(hjust = 0.5)) 
}

cov(datos_precipitacion[,-1])

# Asumiendo que tienes tres gráficos llamados grafico1, grafico2 y grafico3
# Combina los gráficos en un solo cuadro con 1 fila y 3 columnas
grid.arrange(plot_histogram(df, "RMSE", "#00CDCD", "#1E90FF","RMSE", "Histograma de metrica RMSE"),
             plot_histogram(df, "MAE", "#32CD32", "#006400","MAE" ,"Histograma de metrica MAE"),
             plot_histogram(df, "MAPE", "#FFD700", "#FF8C00","MAPE", "Histograma de metrica MAPE"),
             plot_histogram(df, "R2", "#EE82EE", "#9400D3", expression(R^2) ,bquote("Histograma de " ~ R^2)),  
             nrow = 2) #numero de filas

# Dibujamos cada gráfico
plot_histogram(df, "RMSE", "#00CDCD", "#1E90FF", "RMSE")
plot_histogram(df, "MAE", "#32CD32", "#006400", "MAE")
plot_histogram(df, "MAPE", "#FFD700", "#FF8C00", "MAPE")
plot_histogram(df, "R2", "#EE82EE", "#9400D3", "R^2")





