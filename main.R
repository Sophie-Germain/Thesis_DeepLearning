#Series de tiempo de demanda de energía eléctrica 2006-2015
################################ 1. EXPLORACIÓN ################################
library(tseries)

### Cargamos el dataset
write.csv()

elec_consumption=read.csv(file = "consolidado_estaciones.csv", header = TRUE, row.names = 1)
plot(elec_consumption)

new_elec=as.data.frame(elec_consumption[, 1], row.names = row.names(elec_consumption))
names(new_elec) = names(elec_consumption)[1]
plot(ts(new_elec))
write.csv(ts(new_elec), file = "data_noest.csv", row.names=FALSE)

adf.test(diff(log(ts(new_elec))), alternative="stationary", k=0)
plot(diff(log(ts(new_elec))))
write.csv(diff(log(ts(new_elec))), file = "data_est.csv", row.names=FALSE)

acf(diff(log(ts(new_elec))))
pacf(diff(log(ts(new_elec))))

(fit <- arima(log(ts(new_elec)), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 120)))

pred <- predict(fit, n.ahead = 120)

ts.plot(ts(new_elec), 2.718^pred$pred, log = "y", lty = c(1,3))



acf(log(ts(new_elec)))
pacf(log(ts(new_elec)))


----------------------------------------------------------

write.csv(diff(log(AirPassengers)), file = "MyData.csv", row.names=FALSE)
plot(diff(log(AirPassengers)), data=diff(log(AirPassengers)))

AirPassengers_train = ts(AirPassengers[1:115])
length(AirPassengers_train)
AirPassengers_test  = ts(AirPassengers[116:length(AirPassengers)])
length(AirPassengers_test)
length(AirPassengers)

### Graficamos. El número de pasajeros ha incrementado año con año.
plot(AirPassengers_train, data=AirPassengers_train)
plot(AirPassengers_test, data=AirPassengers_test)


### Construimos una regresión lineal
#abline(reg=lm(AirPassengers~time(AirPassengers)), data=AirPassengers)

#Serie acumulada
#plot(aggregate(AirPassengers_train,FUN=mean))

#Box plot sobre los meses nos dirá si hay algún efecto estacional
#La varianza y el promedio mean value en julio y agosto es mayor que en los otros meses.
#La media por mes es diferente, pero la varianza se mantiene pequeña. Esto sugiere un efecto estacional con un ciclo de 12 meses o menos.
#La varianza incrementa con respecto al tiempo
boxplot(AirPassengers_train~cycle(AirPassengers_train))

################################ 2. MODELOS ##################################
#Antes que nada hay que hacer las varianzas constantes. Esto se logra aplicando logaritmo.
#También hay que arreglar la tendencia.Hay que aplicar diferencias para estacionariedad


adf.test(diff(log(AirPassengers_train)), alternative="stationary", k=0)
plot(diff(log(AirPassengers_train)))


#El siguiente paso es ver qué parámetros pondremos en el modelo ARIMA
#d=1 porque sólo necesitamos una diferencia para volver estacionario al modelo
#Para los otros parámetros usamos la grafica de correlación total
acf(log(AirPassengers_train))

#El decaimiento de la ACF es muy lento, lo que significa el proceso no es estacionario
#Pero íbamos a trabajar con diferencias!
acf(diff(log(AirPassengers_train)))

#Función de correlación parcial:
pacf(diff(log(AirPassengers_train)))

#Claramente la ACF curta después del primer lag
#Entonces el valor de p debe ser 0 ya que la ACF es la curva que tiene un corte
#Mientras el valor de q debe ser 1 o 2
#Después de algunas iteraciones encontramos que (p,d,q) = (0,1,1) es la combinación con menor AIC y BIC.
#Ajustemos un modelo ARIMA y hagamos una predicción a 10 años.

(fit <- arima(log(AirPassengers_train), c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12)))

################################ 3. Predicción ################################
pred <- predict(fit, n.ahead = length(AirPassengers_test))

ts.plot(AirPassengers_train, 2.718^pred$pred, log = "y", lty = c(1,3))



##############################################################################
#Función para calcular el error cuadrático medio
rmse <- function(error)
{
    sqrt(mean(error^2))
}

#Función para calcular el error medio absoluto
mae <- function(error)
{
    mean(abs(error))
}


#Calculamos el error en el test set sobre las unidades reales
rmse((2.718^pred$pred) - as.numeric(AirPassengers_test))
mae((2.718^pred$pred) - as.numeric(AirPassengers_test))

#Calculamos el error en el test set sobre las unidades escaladas para compararlo con weka
rmse((pred$pred) - as.numeric(log(AirPassengers_test)))
mae((pred$pred) - as.numeric(log(AirPassengers_test)))


