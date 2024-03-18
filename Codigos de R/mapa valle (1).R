library(maptools)
library(rgeos)
library(rgdal)
library(maps)
library(gpclib)
library(sp)
library(ggplot2)
library(dplyr)
valle <- readOGR(dsn="C:/Users/alexi/OneDrive/Documentos/1 ESTADISTICA/TRABAJO DE GRADO/Datos")
valle.ggmap <- fortify(valle, region= "NOM_MUNICI")

valle.ggmap[valle.ggmap$id=="jamundi",]
table(valle.ggmap$id)
valle.ggmap$id<-chartr("ÁÉÍÓÚáéíóú", "AEIOUaeiou", valle.ggmap$id)
valle.ggmap$id<-tolower(valle.ggmap$id)


DatosPrecip=Datos_climaticos%>%
            group_by(name, month = lubridate::floor_date(month, 'year'))%>% 
            summarise(precipA=sum(precip))
names(DatosPrecip) <- c("id","month","precipA")
DatosPrecip$id<-tolower(DatosPrecip$id)
DatosPrecip$id<-chartr("ÁÉÍÓÚáéíóú", "AEIOUaeiou", DatosPrecip$id)


valle.ggmape <- merge(valle.ggmap, DatosPrecip, by = "id", all = TRUE)
valle.ggmape <- valle.ggmape[order(valle.ggmape$order), ]
valle.ggmape<-valle.ggmape[!is.na(valle.ggmape$precipA),]
table(valle.ggmape$id)


ggplot(data = valle.ggmape, aes(x = long, y = lat, group = group))+
  geom_polygon(aes(fill = precipA))+
  coord_sf(xlim = c(-80, -73), ylim = c(3, 5), expand = FALSE)

  
+facet_wrap(~month)

ggplot(data = valle_sf, aes(x = long, y = lat, group = group))+
  geom_polygon()






sum(Datos_climaticos[Datos_climaticos$name=="Cali","precip"])









