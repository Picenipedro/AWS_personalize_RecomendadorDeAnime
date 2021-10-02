# Recomendador de Anime con Amazon Personalize

!["aws_personalize"](imagen/animes.jpg) 

Para esta Demo utilizo una DataBase de Kaggle con nombre MyAnimeList Database 2020, la cual se encuentra disponible en este link:[https://www.kaggle.com/hernan4444/anime-recommendation-database-2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)

1. [Descarga de Dataset Kaggle](01_Descarga_Dataset.ipynb)
2. [Exploracion y Preparación](02_Exploracion_Prepariacion.ipynb)
3. [Crear las soluciones](03_Creacion_Soluciones.ipynb)
4. [Despliegue de Campañas](04_Desplegando_campanas.ipynb)
5. [Probar Recomendaciones](05_Probando_Recomendaciones.ipynb)
6. [Limpiar el proyecto](06_Clean_Up_Resources.ipynb)

## Introducción

Cada vez los motores de recomendación están cobrando relevancia en los sistemas para personalizar la experiencia del usuario y hacerla más relevante. Ejemplos son la publicidad, recomendaciones en plataforma de Video On Demand, o sugerencias de nuevos contactos en redes sociales. Bien ejecutado un modelo de personalización puede aumentar las ganancias de una empresa o disminuir los costos, pero por el contrario, si se hace mal, termina en una mala experiencia de cliente.

Una de las dificultades de estos proyectos es que requieren conocimientos y experiencia avanzada en Machine Learning (cosa que no todos pueden tener a la mano), por eso comencé a explorar el servicio de Amazon Personalize que brinda de alguna manera un "servicio de motor de recomendaciones con tus propios datos", haciendo todo el proyecto muchísimo más abordable y donde se pueden ver los resultados rápidamente. 

Para entenderlo, armé un proyecto basado en el repositorio [original](https://github.com/aws-samples/amazon-personalize-samples/tree/master/next_steps/workshops/POC_in_a_box),  y vamos a utilizar el servicio de Amazon Personalize para entrenar motores de recomendación de Animes utilizando  las interacciones que han tenido los usuarios con estos animes, como también la otros datos disponibles como Genero y Calificación.


Manos a la obra.


## Primero ¿Que es Amazon Personalize?

- Es un servicio que permite entrenar modelos de personalización y obtener recomendaciones en tiempo real. 

- Basado en la misma tecnología de aprendizaje automatico (ML) utilizada por la tienda www.amazon.com.

- Genera recomendaciones de gran relevancia utilizando técnicas de Deep Learning.

- No requiere tener la experiencia o conocimientos en Machine Learning. 😃 🥳 🎉🎉

Fuente: https://aws.amazon.com/es/personalize/

## Y...¿Como Funciona Amazon Personalize?

!["aws_personalize"](imagen/aws_personalize.png) 

1. Debes proporcionar datos sobre usuarios y elementos para realizar la personalización.

Los datos que utilizamos para modelar en Personalize son de tres tipos:

**Dataset de Interacciones**: este Dataset consiste en el momento en el que usuario interactuó con un ítem, por ejemplo clicks, compras, visualización, calificación. Es el único Dataset obligatorio. 

**Item metadata (opcional)**, el cual contiene información adicional de los ítems, tales como categoría o genero. 

**Dataset de Usuarios (opcional)**, que contiene información de los usuarios como edad o sexo. 

2. Detrás de cámaras, Amazon Personaliza automáticamente:

- Inspecciona los datos
- Identifica lo que es significativo
- Selecciona los algoritmos adecuados
- Entrena y optimiza modelos de personalización

3. Después del entrenamiento se obtiene un modelo entrenado que se puede desplegar para hacer inferencias a través de una API.

Ahora, Nuestro proyecto  Recomendador de Anime.

Los pason para llevar a cabo nuestro recomendador se describen a continuación, pero en resumen son:

1. Crear un DatasetGroup 
2. Creacion de Esquemas y Datasets
3. Importar los datos a los datasets
4. Seleccionar un algoritmo (recipe), Crear una solution y Solution version (modelo entrenado)
5. Crear un endpoint de inferencias (Campaña)
6. Creación de un Event Tracker (para actualizar los DataSets)
7. Opcionales: Creación de Filtros 
8. Resultados
9. Opcional: Rest API


!["aws_personalize"](imagen/creacion.png)

## Paso 1: Crear un DataSet Group

Para crear nuestro proyecto de personalize, primero debemos crear el Dataset Group. Es una especie de grupo donde están todos los recursos (no solo los datasets). Puedes crear varios Dataset Groups y tus modelos no se van topar entre si. 

Por ejemplo, es posible que tenga una aplicación que proporciona recomendaciones para la transmisión de vídeo y otra que recomienda libros de audio. En Amazon Personalize, cada aplicación tendría su propio grupo de conjuntos de datos. 

Y en nuestro GitHub seria estas lineas:

```python 
create_dataset_group_response = personalize.create_dataset_group(
    name = "personalize-anime"
)
dataset_group_arn = create_dataset_group_response['datasetGroupArn']
```
## Paso 2: Creación de un conjunto de datos y un esquema

Una vez que haya completado Paso 1, ya estarás preparado para crear un conjunto de datos. Cuando se crea un conjunto de datos, también se crea un esquema para el conjunto de datos. El esquema de Amazon Personalize sobre la estructura de tus datos y permite a Amazon Personalize analizar los datos.

Para nuestra Demo GitHub seria:

*Interaction DataSet*

Podemos ver que costa de 5 columnas 3 de ellas las requeridas: user_id (que es el ID del usuario), ítem_id (el ID del anime) y timestamp (cuando ocurrio esta evaluación). 

!["aws_personalize"](imagen/interaction.png)

Adicionalmente tenemos valores opcionales que son Event_value (Valor obtenido por el anime) y event_type (que es el tipo del valor, en este caso rating), estos dos últimos son opcionales pero importante para nuestro modelo. 

El esquema se crea con las siguientes lineas de código:


```python
interactions_schema = schema = {
    "type": "record",
    "name": "Interactions",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {
            "name": "USER_ID",
            "type": "string"
        },
        {
            "name": "ITEM_ID",
            "type": "string"
        },
        {
            "name": "TIMESTAMP",
            "type": "long"
        },
        {
            "name": "EVENT_VALUE",
            "type": "float"
        },
        {
            "name": "EVENT_TYPE",
            "type": "string"
        },

    ],
    "version": "1.0"
}

create_schema_response = personalize.create_schema(
    name = "personalize-anime-interactions1",
    schema = json.dumps(interactions_schema)
)
```

*Items DataSet*

En nuestro caso los items corresponde al listado de animes, con su ID (el cual es requerido), estudio que lo creo, año de su lanzamiento, y géneros.

!["aws_personalize"](imagen/items.png)

Los opcionales deben incluir categorical= true, de lo contrario Amazon personalize no utilizara este campo para entrenar el modelo, además estas características las podemos usar para los filtros, que mencionaremos después. 

El esquema se crea con las siguientes lineas de código:


```python
itemmetadata_schema = {
    "type": "record",
    "name": "Items",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {
            "name": "ITEM_ID",
            "type": "string"
        },
        {
            "name": "STUDIOS",
            "type": "string",
            "categorical": True
        },
        {
            "name": "YEAR",
            "type": "int",
        },
        {
            "name": "GENRE",
            "type": "string",
            "categorical": True
        },

        
    ],
    "version": "1.0"
}
```


## Paso 3: Importación de los datos

Ahora estarás listo para importar los datos de entrenamiento a Amazon Personalize. Al importar los datos, puedes optar por importar registros en bloque, importar registros individualmente o ambos, según los requisitos del negocio y la cantidad de datos históricos recopilados. Indistintamente deben estar en un bucket en S3 con permisos para Amazon Personalize.

Después de haber formateado los datos de entrada ([Formatear los datos](https://docs.aws.amazon.com/es_es/personalize/latest/dg/data-prep-formatting.html)) y cargados en S3 ([Cargar a un bucket de S3](https://docs.aws.amazon.com/es_es/personalize/latest/dg/data-prep-upload-s3.html)), debemos importar los datos de forma masiva a Amazon Perzonalice mediante la creación de un trabajo de un import job.

Los import jobs son una herramienta de importación masiva que rellena el DataSet group con los datos del bucket de S3.

En nuestra Demo [GitHub](https://github.com/elizabethfuentes12/AWS_personalize_RecomendadorDeAnime/) creamos el import job:


```python
create_dataset_import_job_response = personalize.create_dataset_import_job(
    jobName = "personalize-anime-item-import",
    datasetArn = items_dataset_arn,
    dataSource = {
        "dataLocation": "s3://{}/{}".format(bucket_name, itemmetadata_filename)
    },
    roleArn = role_arn
)

dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
```

## Paso 4: Creación de la solución

Ahora ya puedes crear una solución. La solución hace referencia a la combinación de una receta de Amazon Perzonalice, parámetros personalizados y una o mas versiones de la solución.  Después de crear una solución con la Version Solution que mejor desempeño tenga, creas la campaña y obtienes las recomendaciones. 

Para crear una solución en Amazon Personalize, debe hacer lo siguiente:

***1- Elija un recipe:*** Un recipe un término de Amazon Personalize que especifica un algoritmo apropiado para entrenar para un caso de uso determinado. 

Los recipe son algoritmos preparados para casos de uso especificos que te permite crear un  sistema de personalización sin experiencia previa en aprendizaje automático.

Amazon Personalize proporciona 3 tipos de recipe:

User personalization: cosiste en un modelo al cual yo le paso un usuario y me entrega un listado de ítems recomendados para ese usuario. 

Aca tenemos un ejemplo de amazon.com donde recomienda productos basados en la historia de compras. 


!["aws_personalize"](imagen/personalization.png)

***Personalized Ranking:*** Le paso un usuario y un listado de ítems y me los entrega ordenados de acuerdo a la preferencia de ese usuario.

Un ejemplo de esto son los productos que te recomienda Amazon en los correos de confirmación de compras. 

!["aws_personalize"](imagen/Personalized_Ranking.png)


***Similar items:*** Se basa en el concepto de filtrado colaborativo, si otro usuario compra el mismo articulo que tu, entonces el modelo te recomienda artículos que ese otro usuario ha comprado. 


!["aws_personalize"](imagen/similar.png)


Más información: [aquí](https://docs.aws.amazon.com/es_es/personalize/latest/dg/working-with-predefined-recipes.html)

Asi cremos el recipe SIMS en nuestara Demo [GitHub](https://github.com/elizabethfuentes12/AWS_personalize_RecomendadorDeAnime/):


```python
sims_create_solution_response = personalize.create_solution(
    name = "personalize-anime-sims",
    datasetGroupArn = dataset_group_arn,
    recipeArn = SIMS_recipe_arn
)
```

Esto se repite para Personalized Ranking y User Personalization.

***2. Creación de una versión de solución (entrene un modelo):***  Entrena el modelo de aprendizaje automático que Amazon Personalize utilizará para generar recomendaciones para tus clientes.

Esto tarde un rato, asi que ten paciencia.

***3. Evaluar la versión de solución:*** Una vez tengamos listas las versiones, debemos evaluar si es la mas adecuada, para eso personalize deja una parte de la data fuera del entrenamiento que usa para probarlo.

Y nos entrega el siguiente cuadro con la evaluación:

!["aws_personalize"](imagen/metricas.png)


*Coverage* es el porcentaje de los items que el modelo está recomendando en general. Es decir por ejemplo en SIMS, tiene un coverage de 45% es decir hay un 55% de items que no están siendo recomendados.

*Precision* 5, 10 y 25 son las precisión de las recomendaciones por ejemplo si recomiendo 10 items y el usuario interactuá con 2, mi precision a 10 es de 20%

*Normalized discount gain* 5,10 y 25 es muy similar a precisión pero acá el orden importa, y hay incentivo por entregar las recomendaciones en orden real de interacción

Si quieres mejorar el coverage debes garantizar que el modelo tenga en cuenta los ítems nuevos y los antiguos con poca preferencia, esto lo puedes modificar seteando el parametro *campaignConfig* al crear la campaña.

Más Información: [https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/personalize.html#Personalize.Client.create_campaign] (https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/personalize.html#Personalize.Client.create_campaign)




## Paso 5: Crear un endpoint de inferencias (Campaña)

Una vez tomada la decisión sobe nuestra Version Solution con mejor desempeño creamos la campaña y la desplegamos. 

Lo hacemos con esta lineas de comandos: 

```python
sims_create_campaign_response = personalize.create_campaign(
    name = "personalize-anime-SIMS",
    solutionVersionArn = sims_solution_version_arn,
    minProvisionedTPS = 1
)
sims_campaign_arn = sims_create_campaign_response['campaignArn']
```

**Paso 6: Creación de un Event Tracker**

Amazon Personalize puede hacer recomendaciones basadas en tiempo real,  solo con data histórica, o con una mezcla de ambos.  Crea el Event Tracker para que Amazon Personalize pueda aprender de la actividad más reciente de su usuario y actualizar las recomendaciones mientras utilizas la aplicación. Esto mantiene tus datos de interacciones actualizados y mejora la relevancia de las recomendaciones de Amazon Personalize.

Lo hacemos con esta lineas de comandos:


```python
response = personalize.create_event_tracker(
    name='AnimeTracker',
    datasetGroupArn=dataset_group_arn
)
```

## Paso 7: Opcional - Filtros 

Ahora que todas las campañas están implementadas y activas, podemos crear filtros. Que te permite filtrar por una condición del ítem dataset, en nuestro caso haremos un filtro para los géneros, también podemos filtrar por campo año o estudio.

```python
createfilter_response = personalize.create_filter(
            name=genre,
            datasetGroupArn=dataset_group_arn,
            filterExpression='INCLUDE ItemID WHERE Items.GENRE IN ("'+ genre +'")'
        )
```

## Paso 8: Resultados

Para ver los resultados trabajamos el notebook [05_Probando_Recomendaciones](https://github.com/elizabethfuentes12/AWS_personalize_RecomendadorDeAnime/blob/main/05_Probando_Recomendaciones.ipynb), donde puedes ver los siguientes resultados.. y además crear los tuyos. 

***Similar Items (SIMS)***

!["aws_personalize"](imagen/sims.png)

***User Personalization***

!["aws_personalize"](imagen/Personalization_ejm.png)

***Usando Filtros***

!["aws_personalize"](imagen/filtros.png)


## Paso 9: Opcional - Rest API  😉

Rest API para POC Amazon Personalize

!["aws_personalize"](imagen/api.png)

El objetivo de este proyecto es un despliegue rápido de una API RESTfull para invocar las campañas de Amazon Personalize. Esto lo haremos con Cloud Development Kit (CDK) en Python.

Este proyecto está tal cual (as-is) para probar las predicciones de Amazon Personalize utilizando API Gateway y Lambda.

[https://github.com/ensamblador/recomendaciones-rest-api](https://github.com/ensamblador/recomendaciones-rest-api)


# Acá el video del evento

[https://www.youtube.com/watch?v=GzdMPwbVfCQ](https://www.youtube.com/watch?v=GzdMPwbVfCQ)


# Fuentes adicionales

Este proyecto esta basado en el repocitorio oficial de AWS:
[https://github.com/aws-samples/amazon-personalize-samples/tree/master/next_steps/workshops/POC_in_a_box](https://github.com/aws-samples/amazon-personalize-samples/tree/master/next_steps/workshops/POC_in_a_box)

Developer Guide : https://docs.aws.amazon.com/es_es/personalize/


