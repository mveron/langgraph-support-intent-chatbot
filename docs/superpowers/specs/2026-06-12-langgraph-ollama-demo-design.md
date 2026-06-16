# Demo de LangGraph con Ollama

## Objetivo

Crear una aplicación educativa en Python que permita explicar a otros
desarrolladores los conceptos principales de LangGraph mediante un ejemplo
pequeño, visible y ejecutable.

La aplicación aceptará preguntas tanto desde una CLI interactiva como desde
una interfaz web con Streamlit. Ambas interfaces usarán el mismo grafo y el
modelo local `qwen3:4b` a través de Ollama.

## Alcance

La demo mostrará:

- Un estado compartido que evoluciona durante la ejecución.
- Nodos con responsabilidades concretas.
- Aristas normales y una arista condicional.
- Dos rutas posibles según la clasificación de la pregunta.
- La ruta y los resultados intermedios de una ejecución.
- Uso del mismo grafo desde dos interfaces diferentes.

No se incluirán herramientas externas, recuperación de documentos, agentes
autónomos, persistencia de conversaciones ni autenticación. Esos elementos
añadirían complejidad sin ayudar a explicar el concepto básico de grafo.

## Flujo del grafo

```text
START
  |
  v
clasificar
  |
  +-- tecnica --> responder_tecnica --+
  |                                   |
  +-- general --> responder_general --+--> END
```

El nodo `clasificar` determina si una pregunta es `tecnica` o `general`. Una
arista condicional usa esa categoría para elegir el siguiente nodo. El nodo
seleccionado genera una respuesta adaptada y finaliza la ejecución.

## Estado compartido

El estado del grafo contendrá:

- `question`: pregunta ingresada por el usuario.
- `category`: clasificación `tecnica` o `general`.
- `answer`: respuesta final.
- `trace`: lista ordenada de eventos legibles de la ejecución.

Cada nodo devolverá únicamente los campos que modifica. LangGraph combinará
esas actualizaciones en el estado que pasa por el grafo.

## Componentes

### Núcleo del grafo

`graph.py` definirá:

- El tipo del estado.
- Los nodos de clasificación y respuesta.
- La función de enrutamiento condicional.
- Una función constructora que compile y devuelva el grafo.
- Una interfaz pequeña para inyectar el modelo de lenguaje.

La inyección permitirá usar Ollama en la aplicación y un modelo controlado en
las pruebas.

### Integración con Ollama

`llm.py` creará la integración de LangChain para Ollama usando el modelo
`qwen3:4b`. Los prompts exigirán una clasificación de formato estable y
respuestas concisas apropiadas para una demo.

Los errores de conexión o la ausencia del modelo producirán mensajes claros
que indiquen verificar que Ollama esté activo y que `qwen3:4b` esté instalado.

### CLI

`cli.py` ejecutará un bucle interactivo:

1. Solicitar una pregunta.
2. Permitir salir con `salir`, `exit` o una entrada vacía.
3. Invocar el grafo.
4. Imprimir la categoría, la ruta recorrida y la respuesta.
5. Capturar errores esperables sin cerrar con un traceback innecesario.

### Interfaz web

`app.py` usará Streamlit y contendrá:

- Título y explicación breve de los conceptos.
- Diagrama permanente del grafo.
- Campo para ingresar una pregunta y botón de ejecución.
- Respuesta final.
- Categoría elegida.
- Ruta recorrida en orden.
- Estado final, visible en formato estructurado.
- Mensajes de error accionables si Ollama no está disponible.

La visualización representará la estructura completa del grafo. Tras una
consulta, la ruta ejecutada se distinguirá visualmente y el registro detallará
qué hizo cada nodo. No se requiere animación en tiempo real: mostrar el
resultado de cada paso conserva la claridad y evita complejidad incidental.

## Flujo de datos

1. La interfaz crea un estado inicial con la pregunta.
2. El grafo llama al nodo clasificador.
3. El clasificador consulta a Ollama y actualiza `category` y `trace`.
4. La arista condicional consulta `category`.
5. El nodo de respuesta elegido consulta a Ollama.
6. El nodo actualiza `answer` y `trace`.
7. La interfaz representa el estado final.

## Manejo de errores

- Una pregunta vacía no invocará el grafo.
- Una salida inesperada del clasificador se normalizará a `general`.
- Los fallos al conectar con Ollama se traducirán a una explicación breve.
- La web conservará la interfaz utilizable después de un error.
- La CLI volverá a solicitar una pregunta después de un error recuperable.

## Pruebas

Las pruebas usarán `pytest` y un LLM falso determinista. Cubrirán:

- Enrutamiento de preguntas técnicas.
- Enrutamiento de preguntas generales.
- Normalización de una clasificación inesperada.
- Contenido y orden de la traza.
- Resultado final de ambas ramas.

No se exigirá que Ollama esté activo para ejecutar las pruebas unitarias. La
verificación manual final sí ejecutará una pregunta real contra `qwen3:4b` si
el servicio local está disponible.

## Estructura prevista

```text
.
|-- app.py
|-- cli.py
|-- graph.py
|-- llm.py
|-- requirements.txt
|-- README.md
|-- tests/
|   `-- test_graph.py
`-- docs/
    `-- superpowers/
        `-- specs/
            `-- 2026-06-12-langgraph-ollama-demo-design.md
```

## Criterios de aceptación

- `python cli.py` inicia una sesión interactiva y procesa varias preguntas.
- `streamlit run app.py` muestra el grafo y permite ejecutar preguntas.
- Ambas interfaces usan exactamente el mismo grafo compilado.
- La ruta técnica y la ruta general pueden demostrarse.
- La interfaz web muestra categoría, traza, estado y respuesta.
- La aplicación usa Ollama con `qwen3:4b`.
- Las pruebas unitarias pasan sin depender de Ollama.
- El README permite instalar y ejecutar el proyecto desde PowerShell.
