# Agente RL para CPMP with A2C

Trabajo de investigación y desarrollo para Practica I de Ingeniería Civil Informática.

## Introducción


### Agente 
Un agente es un algoritmo que interactúa con un ambiente a traves de acciones, transitando desde un estado actual a uno nuevo recibiendo una recompensas asociada.
El estado corresponde a "la situación actual" del ambiente y entrega toda la información necesaria para que pueda tomar decisiones.

### Ambiente
Para nuestro problema en particular, está ideado para el "Container Pre-marshalling Problem", lo que significa que nuestro ambiente será una serie de pilas con contenedores en ellas.

![Figura 1: Escenario con 7 containers](img/fig1.png)

<p align=center> Figura 1: Escenario con 7 containers </p>

### Aprendizaje reforzado

### Deep RL

## Marco teórico
La falta de documentación de esta estrategia, ha motivado la investigación y desarrollo de este proyecto. Se ha investigado la utilización del algoritmo de Greedy para la resolución del CPMP.

### Modelado del agente
Se diseñará un agente que aprenda mediante recompensas, para ello se utilizará
la ecuación de bellman como politica de elección de acciones.

#### Estados
Para modelar los estados se utilizó una lista de pilas (stacks). Por ejemplo, la representación de la **Figura 1** sería

    [[1], [1], [4, 6, 5], [7, 3]]

#### Acciones
Las acciones son representadas por una tupla de dos elementos, siendo el primero el stack de origen y en segundo, el stack de destino

    (0, 3) : Se mueve container de la primera pila a la cuarta

#### Recompensa
Para las recompensas, tenemos dos tipos:
* Recompensa inmediata: esta se atribuye al movimiento de un container, 
                        se valoriza como -1 
* Recompensa acumulada: esta corresponde a la suma de las recompensas desde un 
                        estado hasta el estado final

## Objetivos
El principal objetivo de este trabajo es entrenar un agente mediante un aprendizaje de refuerzo para resolver el  problema del CPMP. Para ello se investiga y modela la estrategia de actor-critico para el aprendizaje.

## Programa
El programa principal corresponde al archivo [main.py](main.py) en este se inicializan los datos, se crea en ambiente (escenario inicial) y se comienza el entrenamiento:
1. Actor escoge una acción
2. Ambiente realiza el actor generando:
  - un nuevo estado
  - la recompensa
  - verifica si se resolvió el problema (variable _done_)
3. Critico asocia (_recuerda_) el estado actual más la acción realizada, la recompensa ganada, el nuevo estado generado y si esto significó la resolución del problema.
4. Se entrena el actor y el critico
5. Se actualiza el estado actual como el nuevo generado
6. Se repiten los pasos hasta que el problema este resuelto

## Layout

## Actor Critic
