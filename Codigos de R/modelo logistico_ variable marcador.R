###---Codigo de modelo logistico aplicado a la naturaleza de los datos faltante-------###

#Lectura de librerias
library("easypackages")


lib_req<-c("readr","ggplot2","tidyverse","MASS","visdat","corrplot","plotrix","doBy","FactoMineR","factoextra","caret","e1071","pROC","class","rpart","rpart.plot","randomForest")# Listado de librerias requeridas por el script
easypackages::packages(lib_req)         # Verificación, instalación y carga de librerias.

#Lectura de datos en formato para modelo
datos_precipitacion <-read.csv("Datos/datos_seleccionados_para_modelo_coordenadas.txt",na.strings = c("N/A", " ", "NA"),sep = ";")

#Estructura de los datos
str(datos_precipitacion)

#Adicion de variable condicional con faltante para modelo 
datos_precipitacion$marcador_faltante<-as.numeric(is.na(datos_precipitacion$prec))

str(datos_precipitacion)


#Mejorar esta configuracion
datos_precipitacion=transform(datos_precipitacion, marcador_faltante=factor(marcador_faltante,levels=0:1,labels=c("Completo","Faltante")))

table(datos_precipitacion$marcador_faltante)

#tasa de faltantes:
# Usa ggplot2 para hacer el gráfico de barras
ggplot(datos_precipitacion, aes(x = marcador_faltante, y = Frequency, fill = marcador_faltante)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Grafico de Barras de la variable marcador", x = "Marcador", y = "Frecuencia") +
  scale_fill_discrete(name = "Marcador") +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))  # Centrar el título

freq_table <- table(datos_precipitacion$marcador_faltante)

# Convierte la tabla en un data frame para ggplot2
df <- data.frame("marcador_faltante" = names(freq_table), "Frequency" = as.vector(freq_table))




#


# Usa ggplot2 para hacer el gráfico de barras
ggplot(df, aes(x = marcador_faltante, y = Frequency, fill = marcador_faltante)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Grafico de Barras de la variable marcador_faltante", x = "marcador_faltante", y = "Frecuencia") +
  scale_fill_discrete(name = "marcador_faltante") +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))  # Centrar el título

####  División de datos entrenamiento/test                               ####
#------------------------------------------------------------------------------#

N = nrow(datos_precipitacion); id = 1:N; N.tr=ceiling(0.8*N)
set.seed(10)
id.tr=sample(id,N.tr,replace=F)
datos_precipitacion.tr= datos_precipitacion[id.tr,]   # Datos de entrenamiento de los modelos
datos_precipitacion.te= datos_precipitacion[-id.tr,]  # Datos para evaluar el performance de los modelos


###--------Lectura de datos para prueba de Little------------------### 
str(datos_precipitacion)

library(BaylorEdPsych)
resultado <- mcar_test(tu_dataframe)
print(resultado)

library(naniar)
resultado <- test_mcar(datos_precipitacion)
print(resultado)


install.packages("MissMech")
library(MissMech)
resultado <- TestMCARNormality(data = tu_dataframe)
print(resultado)

count(datos_precipitacion, na.rm = TRUE)

mcar_test(datos_precipitacion)

#El por que se generan los datos faltantes - Daños de la estacion y maquina
#no esta asociado a la variable de respuesta - Explicar la probabilidad de las variables
#para observar el comportamiento de los datos faltantes

## Cual es el proposito de los datos, si se va predecir un modelo que ajuste 
## las tecnicas de estimaciones - verosimilitud asume las distribucioens
## 

## simple completamente aletorio - no hay dependencia con lo que genera los datos faltantes
# explicacion por los datos observados 

## mixto aletorio - 
##Dependencia del tiempo con respecto al dia anterior



## complejo no aletorio - depende de datos que no se observan 

#-----------------------Tareas--------------------#
# ajustar el modelo
# corregire la variable ID
# adelantar la parte introductoria 
# justificacion 
# Deficiones en el marco teorico
# metodologia - conectar el marco con lo que nosostros hicimos para los datos nuestros,
#descibiendo las covariables, como tal las herramientas respecto a los desarrolos
# 



library(alr4)
data("lathe1")









