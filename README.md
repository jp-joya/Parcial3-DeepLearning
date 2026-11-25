# Parcial 3 — UCI HAR: RNN, LSTM y CNN→LSTM

Proyecto para ejecutar una comparación reproducible durante la noche y producir un notebook documentado y un informe IEEE de máximo dos páginas con resultados reales.

## Ejecutar esta noche

El entorno `.venv` y el dataset se prepararon en este equipo. Desde esta carpeta:

```bash
bash scripts/start_night.sh 8
```

Ejecuta en segundo plano hasta aproximadamente ocho horas. Se puede cerrar la terminal. **El equipo debe permanecer encendido, conectado y sin suspensión automática.** El entrenamiento no necesita Internet después de instalar dependencias y descargar los datos; la primera compilación con Tectonic puede descargar paquetes LaTeX. El presupuesto se verifica entre épocas: puede excederse por una época y por la evaluación/análisis finales. La descarga inicial no cuenta dentro del presupuesto.

```bash
tail -f runs/overnight/console.log
PYTHONPATH=src .venv/bin/python -m har.experiment status
```

`Ctrl+C` en `tail` solo cierra la vista del log. Para correr en primer plano: `bash scripts/run_night.sh 8`.

Si se agota el presupuesto, se devuelve código 2 y se conserva el checkpoint. **Repite el mismo comando para continuar** con un nuevo presupuesto. Una vez fijada la selección en `selection.json`, la reanudación continúa las semillas finales y no abre más búsqueda. Si no se completó una configuración de las tres arquitecturas, continúa la búsqueda en la próxima ejecución. No se mezcla una ejecución con cambios de código, configuración, versiones o dispositivo: para otro experimento usa otra salida, por ejemplo `bash scripts/start_night.sh 8 runs/experimento2`.

## Qué se va a entrenar

- Un único dataset, nueve canales temporales de `Inertial Signals`; no se usan las 561 características de `X_train.txt`/`X_test.txt`.
- 21 sujetos oficiales de entrenamiento: **16 para train y 5 para dev**. Dev prefijado: **5, 16, 21, 28, 30**. Se conservan los **9 sujetos oficiales de test**.
- Misma adquisición y clases en las particiones, pero personas diferentes. Esto reduce la fuga por ventanas solapadas; no demuestra igualdad exacta de distribuciones.
- **F1 macro** es la métrica de selección. Accuracy, F1 por clase y desempeño por sujeto son diagnósticos secundarios.
- Baselines: clase mayoritaria y regresión logística sobre 36 estadísticas de las señales (media/desviación/mínimo/máximo por canal). Se ajustan y evalúan en dev antes de la búsqueda.
- **16 configuraciones × 3 arquitecturas = 48 búsquedas**, con semilla 42. Cada configuración se comparte entre los tres modelos. El primer conjunto es una referencia prefijada y los otros se muestrean sin repetición.
- Se repite la mejor configuración **de cada arquitectura** con tres semillas nuevas: 101, 202 y 303. **9 entrenamientos finales; hasta 57 en total.** No se selecciona la mejor semilla usando test, ni se forma un ensemble.
- Se reserva el 70% del presupuesto para búsqueda. Si se agota, solo se selecciona entre las configuraciones completadas por las tres arquitecturas. Se rota el orden de las arquitecturas para reducir efectos del corte de tiempo. Los ensayos incompletos se guardan, pero quedan excluidos si ya se cerró la selección.
- Checkpoint por época, early stopping (paciencia 10, máximo 60 épocas), AdamW y clipping del gradiente. El mejor checkpoint se elige por F1 macro dev con mejora mínima de 0.0001.
- Las semillas finales siguen usando los mismos train/dev; **no** se incorporan datos dev al ajuste. Solo después de fijar todos sus checkpoints se leen señales/etiquetas de test para evaluar los modelos y baselines. La evaluación se guarda y se reutiliza al reanudar.

### Espacio de hiperparámetros

| Parámetro | Valores |
|---|---|
| Contexto temporal | 64, 128 muestras |
| Unidades recurrentes | 32, 64, 128 |
| Dropout | 0, 0.25, 0.50 |
| Tasa de aprendizaje | 0.0003, 0.001, 0.003 |
| Weight decay | 0, 0.0001, 0.001 |

Son 162 combinaciones posibles, de las cuales se prueban 16. Fijos: una capa recurrente, batch 128, convolución de 32 canales/kernel 5 con pooling 2. RNN usa tanh; LSTM usa sus compuertas habituales. CNN-LSTM: Conv1D→ReLU→MaxPool→Dropout→LSTM→Dropout→Linear. En todos los casos se usa el último estado oculto y seis logits para entropía cruzada.

La normalización por canal se ajusta **solo con train**, después de recortar el contexto elegido. Para 64 muestras se usa el sufijo de cada ventana original de 128: no se reconstruyen grabaciones continuas y no se crean nuevos ejemplos. Los baselines usan 128 muestras; se documenta esta diferencia de contexto. Igualdad de configuraciones no implica igualdad del número de parámetros o del tiempo de cómputo.

La duración real depende del equipo y de la parada temprana; no se mantiene trabajo artificialmente hasta completar ocho horas. El entorno local usa PyTorch CPU y cuatro hilos para evitar sobresuscripción. Se solicitan operaciones deterministas; algunas operaciones/dispositivos pueden no garantizar determinismo estricto. Las versiones quedan fijadas en `uv.lock`; los manifiestos registran versiones, configuración, dispositivo y hashes del código y del ZIP.

## Al día siguiente

Si el proceso completó todo, encontrarás:

| Archivo/carpeta | Contenido |
|---|---|
| `runs/overnight/analysis/summary.csv` | Medias y desviaciones entre semillas |
| `analysis/per_seed_metrics.csv` | Train/dev/test, parámetros, épocas y tiempo por semilla |
| `analysis/search_results.csv` | Hiperparámetros y resultados de todos los ensayos terminados |
| `analysis/learning_curves.pdf` | Curvas de pérdida y F1 de los tres modelos y todas las semillas |
| `analysis/confusion_test.pdf` | Matrices normalizadas de la primera semilla prefijada |
| `analysis/by_class.csv`, `by_subject.csv` | Desglose de errores de dev y test |
| `analysis/errors_*_dev.csv` | Errores de validación ordenados por confianza |
| `analysis/error_overlap.csv` | Errores compartidos/exclusivos entre modelos |
| `analysis/diagnostics.json` | Brechas train/dev y recomendación por validación |
| `analysis/report/main.tex` | Borrador IEEE con métricas y tablas reales |
| `analysis/informe_ieee.zip` | Paquete listo para subir a Overleaf |
| `final/*/best.pt` | Pesos, normalización, arquitectura e hiperparámetros |

Las rutas abreviadas de la tabla están dentro de `runs/overnight/`. Si falta `evaluation.json`, consulta el estado y reanuda; el informe no se genera con datos parciales.

Abre el notebook:

```bash
.venv/bin/jupyter lab notebooks/Parcial3_UCI_HAR.ipynb
```

Ejecuta sus celdas para incorporar las tablas y figuras a la entrega. La ejecución larga está desactivada dentro del notebook por defecto; usa los scripts de la noche. El notebook contiene guía de análisis y defensa, y usa el entorno de Python que lo lanza.

### Informe IEEE

`report/template.tex` es la fuente; el análisis crea un paquete autocontenido con clase **IEEEtran**, tablas, figuras y texto calculado a partir de la ejecución. **Edita nombre/universidad y revisa la interpretación de errores antes de entregar.** Es un borrador basado en evidencia, no una explicación causal automática de cada error.

1. Sube `runs/overnight/analysis/informe_ieee.zip` a Overleaf y selecciona `main.tex` como documento principal (pdfLaTeX).
2. O compila localmente (este equipo tiene Tectonic en `.tools/tectonic`; en una instalación nueva, instala TeX Live/Tectonic):

```bash
bash scripts/build_report.sh
```

El script usa `pdfinfo` si está disponible y falla si el PDF supera dos páginas. El texto/fuentes no deben reducirse artificialmente: sintetiza el contenido si tus ediciones superan el límite. El ZIP se crea antes de las ediciones manuales: si editas el informe generado, conserva esos cambios; **volver a ejecutar `analyze` regenera y sobrescribe el borrador**. El notebook solo lee artefactos y no lo regenera.

```bash
# Regenerar análisis y borrador desde los resultados guardados (sin entrenar):
PYTHONPATH=src .venv/bin/python -m har.experiment analyze --out runs/overnight
```

Las métricas de una prueba de humo aparecen marcadas explícitamente y **no sirven para concluir qué arquitectura gana**. `runs/smoke*` no se usa para el informe del experimento nocturno.

## Instalación desde cero y verificación

Requiere Linux (el bloqueo de ejecución usa `fcntl`), Python 3.11–3.13, `uv`, aproximadamente 2 GB para el entorno y 1 GB adicional para datos/artefactos. La instalación predeterminada obtiene PyTorch CPU desde su índice oficial. El auto-selector de dispositivos también admite una instalación compatible con CUDA/MPS, pero cambiar ese entorno requiere una salida nueva y volver a validar dependencias; no está configurado como instalación GPU predeterminada.

```bash
uv sync --locked
PYTHONPATH=src .venv/bin/python -m har.experiment prepare
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m har.experiment run \
  --config configs/smoke.json --out runs/smoke --hours 0.25
```

La prueba de humo usa un subconjunto estratificado de hasta 192 ejemplos por partición, dos épocas y una semilla final. Verifica descarga/carga, búsqueda, entrenamiento, predicciones y exportación del informe. Las pruebas automatizadas verifican separación por sujeto, normalización/orden temporal, gradientes de las tres arquitecturas, planificación reproducible, presupuesto y recuperación exacta tras interrupción en CPU.

## Cómo defender las decisiones

- **Ng, capítulos 5–12:** métrica única, conjuntos dev/test definidos antes de iterar y test reservado. Todos los ajustes usan dev.
- **Capítulos 14–19:** inspección de ejemplos errados; categorías de confusión, actividad y sujeto. Cualquier nueva decisión surgida al observar test requiere otra evaluación externa.
- **Capítulos 20–32:** contrastar train/dev y curvas. Una brecha no identifica por sí sola la causa: cambia el sujeto, y no existe aquí una referencia humana/óptima para cuantificar sesgo evitable. La regresión logística es una referencia práctica, no el error de Bayes.
- **Regularización:** dropout explícito después del estado recurrente (el dropout interno de LSTM no actúa con una sola capa), weight decay y parada temprana.
- **Límites:** una sola partición dev, dependencia entre ventanas del mismo sujeto, número de parámetros distinto y desviación entre semillas que no equivale a un intervalo de confianza sobre personas nuevas.
- **Preguntas clave:** ¿por qué F1 macro?, ¿por qué no dividir ventanas al azar?, ¿qué aporta la memoria LSTM?, ¿por qué la convolución puede ayudar o perder información?, ¿qué evidencia sustenta recomendar un modelo?, ¿qué harías si la referencia logística gana?

## Fuentes

- [UCI HAR: descripción, licencia y DOI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0. La descarga registra SHA-256 local y verifica que no cambie al reutilizar el archivo; esto detecta modificaciones posteriores, no sustituye un checksum firmado por el proveedor.
- Andrew Ng, *Machine Learning Yearning*, borrador 2018, PDF incluido en la carpeta original.
- [PyTorch: reproducibilidad](https://docs.pytorch.org/docs/stable/notes/randomness.html) y [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
- [IEEEtran en CTAN](https://ctan.org/pkg/ieeetran). Se conserva la licencia en la cabecera de `report/IEEEtran.cls`.
