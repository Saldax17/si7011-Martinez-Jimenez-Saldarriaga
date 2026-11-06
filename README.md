# SI7011-Martinez-Jimenez-Saldarriaga
SI7011 · Evento evaluativo 1 · Aprendizaje y entrenamiento (S01–S02)
SI7011 · Evento evaluativo 2 · Arquitecturas para visión (S03)

## Evento evaluativo 1 · Avance por integrante

### Integrante 1 — Santiago (Datos y línea base)
- TODO 1 (partición estratificada 60/20/20, submuestra de 50 000 filas y `StandardScaler` ajustado solo con entrenamiento sobre las 10 numéricas), corrida `1-linea-base` y diagnóstico inicial · `evento1/sesion_02_6_integrador_covertype.ipynb` · §2 TODO 1 y §4 · corrida `1-linea-base` · best_val_loss = 1.1963, val_acc = 0.488, F1 macro = 0.094: predice la clase 2 en el 100 % de los casos, es decir, subajuste por optimización (commit `a041301`).

### Integrante 2 — Sara Martínez (Modelo y predicciones)
- `my_model()`: MLP explícito de 8×256 (`Linear → BatchNorm1d → ReLU`) con inicialización He (`kaiming_normal_`, sesgos en 0). Tiene la misma forma que la línea base para aislar el efecto de la init y de BatchNorm · `evento1/sesion_02_6_integrador_covertype.ipynb` · §5 TODO 2 · verificación: pérdida inicial = 1.810 (ln 7 = 1.946) y sobreajuste de 64 ejemplos con pérdida = 0.0002.
- Sin Dropout, porque la línea base subajusta: train_loss = 1.1985 ≈ val_loss = 1.1963 (brecha ≈ 0.002) · §5, celda markdown después de la verificación · corrida `1-linea-base`, mejor época.
- Pendiente: predicción y corrida `2-mi-receta` (necesitan `my_opt`, TODO 3) y TODO 4 `predict_raw`.

### Integrante 3 — Optimización y experimentación (pendiente)
