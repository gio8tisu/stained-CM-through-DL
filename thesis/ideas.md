A continuación se presenta una serie de ideas sobre futuros pasos en el modelo de transformación CM->H&E.

## Red _despeckling_
Experimentos:
* Dataloader con "contaminacion" gaussiana o gamma (entrenar con imagenes de fluorescencia)
* skip-connection aditiva
* log() -> skip-connection aditiva -> exp()
* skip-connection multiplicativa
* entrenar con adversarial loss

## Incluir filtrado de ruido en red generativa
En vez de entrenar y aplicar las dos redes (_despeckling NN_ y _stain NN_) por separado,
se propone incluir la arquitectura de la red _despeckling_ en el modelo generativo para así "aprender" conjuntamente
los parámetros de ambos modelos.  
Experimentos:
* Antes del _encoder_
* "Dentro" de la U-net
 
## Variar funcion de perdida de la _cyclicGAN_

La función de pérdida de la _cyclicGAN_ está compuesta de varios tipos de perdidas
-_adversarial loss_, _cycle loss_, _identity loss_ (para cada generador)-. La idea es simplemente ver que efecto tienen
sobre el resultado final cada una de ellas haciendo variar el peso que tienen dentro de la función de périda (_lambdas_)
especialmente la _identity loss_.

## Cambiar arquitectura de redes generativas por U-net
La arquitectura actual de las GANs es un _encoder-decoder_ simple con una serie de bloques "residuales" en el cuello de
botella. Para lograr una imagen de salida que conserve más los componentes de la imagen de entrada, se pueden conectar
las capas del codificador con las del decodificador. Una manera de hacer esto (la que hace U-net), es concatenando sobre
el eje de los canales los _feature maps_ de cada capa del codificador con las simétricas en el decodificador.
Otra manera de conectar las dos "partes" del modelo es simplemente sumando los mapas de características.

## Segmentacion como entrada adicional en el modelo
La transformación que se debe aplicar a las imagenes de entrada es muy distinta dependiendo de la región. Para incentivar
esa diferenciación, se plantea hacer una segmentación de la imagen previa a la entrada del modelo y añadir esa información
(imagen segmentada) como un nuevo canal de la imagen de entrada. Esta segmentación se puede hacer con métodos "tradicionales"
no supervisado como _clustering_, contornos activos, _watershed_... o con redes neuronales convolucionales,
pero este último requeriria datos etiquetados costosos de obtener.  
EDIT: una segmentacion "no supervisada" no asocia etiquetas "consistentes" a los pixeles.

## Modelo con varias salidas para distintos tipos de tinción
Las imágenes de CM tienen información que no existe en la tinción H&E por lo tanto se pierde información en la
transformación de CM a H&E. Para evitar esto (?), se propone trabajar con varias tinciones con información complementaria
de manera que para una imagen de entrada tendriamos N salidas (una para cada tipo de tinción) -de hecho podria ser una
única salida con más canales, por ejemplo si se trabaja con 3 tipos de tinción (imagenes RGB) la salida tendira 9 canales-.  
EDIT: esto supondria disponer de suficientes muestras de los distintos tipos de tinción, cosa que es costosa.

## Encoder -> clustering -> conditional gan
