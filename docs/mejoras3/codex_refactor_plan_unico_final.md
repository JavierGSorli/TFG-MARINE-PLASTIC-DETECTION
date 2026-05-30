# Refactorización final coherente del proyecto TFG — Plan único para Codex

## Objetivo

Reorganizar y refactorizar el proyecto para que tenga una pipeline clara, intuitiva, reproducible y fácil de explicar en el TFG.

La estructura final debe seguir esta idea:

```text
scripts/   -> scripts ejecutables numerados en orden lógico
outputs/   -> salidas equivalentes a cada script/fase
src/       -> funciones reutilizables, no scripts de pipeline
reports/   -> tablas, figuras y ejemplos finales para la memoria
archive/   -> scripts antiguos, outputs legacy, debug y documentación vieja
```

El proyecto actual contiene scripts históricos, scripts nuevos, outputs mezclados y varios scripts que podrían fusionarse o reordenarse. La prioridad es dejarlo más limpio y coherente **sin cambiar la lógica científica de los resultados salvo que se indique explícitamente**.

---

# Principios obligatorios

1. No borrar definitivamente archivos antiguos.  
   Todo lo sustituido debe moverse a:

```text
archive/deprecated_scripts/
archive/deprecated_outputs/
archive/debug/
archive/old_docs/
```

2. No dejar scripts ejecutables dentro de `src/`.

3. Todos los scripts ejecutables deben estar en `scripts/`.

4. Todos los scripts deben poder ejecutarse desde la raíz del proyecto:

```bash
python scripts/XX_nombre.py
```

5. Todos los scripts deben usar rutas centralizadas desde:

```text
src/common/config.py
```

o

```text
src/common/paths.py
```

6. No usar rutas absolutas tipo:

```python
C:\CDIA_oficial\tfg\...
```

7. No usar imports antiguos como:

```python
from config import ...
from pipeline_utils import ...
from geo_utils import ...
from model_validation_utils import ...
```

Usar imports desde `src`, por ejemplo:

```python
from src.common.config import OUTPUTS_DIR, REPORTS_DIR
from src.evaluation.grouped_validation import ...
```

8. La validación agrupada por fecha solo se aplica a modelos entrenados con el dataset propio:

```text
XGBoost
modelos clásicos
stacking
calibración aprendida
fine-tuning
```

No se aplica a modelos preentrenados MARIDA ni a índices espectrales, porque no se entrenan con este dataset.

9. Mantener explícitamente la diferencia entre:

```text
XGBoost exploratorio -> puede tener leakage temporal
XGBoost grouped-by-date -> resultado metodológicamente defendible
```

10. La visualización debe ir al final de la pipeline y debe conservar el comportamiento útil del antiguo `11_visualizacion.py`.

11. Generar un reporte de migración:

```text
reports/refactor_report.md
```

---

# Estructura final deseada

```text
tfg-marine-plastic-detection/

├── README.md
├── environment.yml
├── .gitignore
│
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_explore_nature_candidates.py
│   ├── 02_download_dataset.py
│   ├── 03_build_dataset_metadata.py
│   ├── 04_annotate_patch_quality.py
│   ├── 05_run_unet_marida.py
│   ├── 06_run_rf_marida.py
│   ├── 07_run_resnet_marida.py
│   ├── 08_run_spectral_indices.py
│   ├── 09_unify_predictions.py
│   ├── 10_evaluate_pretrained_baselines.py
│   ├── 11_error_analysis_baselines.py
│   ├── 12_analysis_by_date.py
│   ├── 13_build_grouped_splits.py
│   ├── 14_run_xgboost_experiments.py
│   ├── 15_compute_sam_features.py
│   ├── 16_compute_glcm_features.py
│   ├── 17_build_feature_table_v2.py
│   ├── 18_train_classical_models_grouped.py
│   ├── 19_run_ablation_grouped.py
│   ├── 20_run_external_models.py
│   ├── 21_evaluate_external_models.py
│   ├── 22_visualize_predictions.py
│   └── 23_export_reports.py
│
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── paths.py
│   │   └── io.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── metadata.py
│   │   ├── patch_io.py
│   │   └── mask_utils.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── indices.py
│   │   ├── sam.py
│   │   ├── glcm.py
│   │   └── feature_table.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── marida_unet.py
│   │   ├── marida_rf.py
│   │   ├── marida_resnet.py
│   │   ├── xgboost_models.py
│   │   └── classical.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── grouped_validation.py
│   │   ├── error_analysis.py
│   │   └── ablation.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── rgb.py
│       ├── overlays.py
│       ├── galleries.py
│       └── maps.py
│
├── outputs/
│   ├── 00_check_environment/
│   ├── 01_explore_nature_candidates/
│   ├── 02_download_dataset/
│   ├── 03_build_dataset_metadata/
│   ├── 04_annotate_patch_quality/
│   ├── 05_run_unet_marida/
│   ├── 06_run_rf_marida/
│   ├── 07_run_resnet_marida/
│   ├── 08_run_spectral_indices/
│   ├── 09_unify_predictions/
│   ├── 10_evaluate_pretrained_baselines/
│   ├── 11_error_analysis_baselines/
│   ├── 12_analysis_by_date/
│   ├── 13_build_grouped_splits/
│   ├── 14_run_xgboost_experiments/
│   ├── 15_compute_sam_features/
│   ├── 16_compute_glcm_features/
│   ├── 17_build_feature_table_v2/
│   ├── 18_train_classical_models_grouped/
│   ├── 19_run_ablation_grouped/
│   ├── 20_run_external_models/
│   ├── 21_evaluate_external_models/
│   ├── 22_visualize_predictions/
│   └── 23_export_reports/
│
├── reports/
│   ├── tables/
│   ├── figures/
│   ├── visual_examples/
│   ├── methodology/
│   ├── refactor_report.md
│   ├── pipeline_summary.md
│   └── final_results_index.md
│
└── archive/
    ├── deprecated_scripts/
    ├── deprecated_outputs/
    ├── debug/
    └── old_docs/
```

---

# Decisión sobre XGBoost

## Pregunta

Antes se proponía:

```text
14_train_xgboost_exploratory.py
15_train_xgboost_grouped.py
16_compare_xgboost_exploratory_vs_grouped.py
```

Pero puede simplificarse.

## Decisión recomendada

Unificar esos tres scripts en uno solo:

```text
scripts/14_run_xgboost_experiments.py
```

Razón:

- los tres forman parte del mismo bloque lógico;
- evita tener demasiados scripts;
- permite comparar automáticamente XGBoost exploratorio vs grouped;
- mantiene todo lo relacionado con XGBoost en una única carpeta de output.

Este script debe soportar modos:

```bash
python scripts/14_run_xgboost_experiments.py --mode exploratory
python scripts/14_run_xgboost_experiments.py --mode grouped
python scripts/14_run_xgboost_experiments.py --mode compare
python scripts/14_run_xgboost_experiments.py --mode all
```

También aceptar alias:

```bash
python scripts/14_run_xgboost_experiments.py --mode both
```

equivalente a:

```text
exploratory + grouped + compare
```

## Outputs de XGBoost

Todo debe guardarse bajo:

```text
outputs/14_run_xgboost_experiments/
```

con subcarpetas:

```text
outputs/14_run_xgboost_experiments/
├── exploratory/
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── model.joblib
│   └── xgboost_exploratory_summary.md
│
├── grouped/
│   ├── fold_metrics.csv
│   ├── out_of_fold_predictions.csv
│   ├── summary_metrics.csv
│   ├── model_config.json
│   └── xgboost_grouped_summary.md
│
└── comparison/
    ├── xgboost_comparison.csv
    └── xgboost_comparison_summary.md
```

## Advertencia obligatoria

En `xgboost_exploratory_summary.md` incluir:

```text
Este resultado es exploratorio y puede estar afectado por leakage temporal, ya que no separa patches por fecha/producto. 
Se conserva únicamente como comparación frente a la validación grouped-by-date.
```

En `xgboost_grouped_summary.md` incluir:

```text
Este resultado usa validación agrupada por fecha y debe considerarse la evaluación metodológicamente principal para XGBoost.
```

En `xgboost_comparison_summary.md` explicar:

```text
Si el rendimiento grouped es menor que el exploratorio, esto puede indicar que parte del rendimiento inicial dependía de similitudes temporales entre patches.
```

## Archivar scripts antiguos de XGBoost

Mover a `archive/deprecated_scripts/` si existen:

```text
08_train_xgboost.py
19_train_xgboost_grouped_cv.py
20_compare_xgboost_exploratory_vs_grouped.py
```

El nuevo `14_run_xgboost_experiments.py` debe reutilizar su lógica.

---

# Orden final de scripts y función de cada uno

## 00 — `00_check_environment.py`

Crear si no existe.

Debe:

- comprobar estructura del proyecto;
- comprobar que `src` es importable;
- comprobar dependencias principales;
- comprobar que no hay scripts ejecutables dentro de `src`;
- comprobar que no quedan imports antiguos;
- comprobar que no hay rutas absolutas sospechosas;
- comprobar que cada script tiene carpeta equivalente en `outputs/`.

Salida:

```text
outputs/00_check_environment/environment_check.md
```

---

## 01 — `01_explore_nature_candidates.py`

Renombrar desde:

```text
00_explore_candidates.py
```

Objetivo:

- explorar candidatos del catálogo Nature;
- filtrar por zona del Estrecho/Gibraltar;
- generar CSV de candidatos.

Salida:

```text
outputs/01_explore_nature_candidates/gibraltar_candidates.csv
```

---

## 02 — `02_download_dataset.py`

Renombrar desde:

```text
01_download_dataset.py
```

Objetivo:

- generar/descargar patches Sentinel-2;
- positivos SI;
- negativos NO;
- pseudo-máscaras Nature-derived;
- resumen de generación.

Salida:

```text
outputs/02_download_dataset/
```

No implementar modos complejos de negativos. Mantener la lógica actual, pero con rutas nuevas y resumen claro.

---

## 03 — `03_build_dataset_metadata.py`

Fusionar:

```text
12_build_dataset_metadata.py
17_build_group_ids.py
```

Objetivo:

- crear `dataset_metadata.csv`;
- crear `dataset_metadata_with_groups.csv`;
- añadir `group_id = date`;
- añadir columnas manuales vacías para calidad/anotación.

Salidas:

```text
outputs/03_build_dataset_metadata/dataset_metadata.csv
outputs/03_build_dataset_metadata/dataset_metadata_with_groups.csv
```

Columnas mínimas:

```text
patch
date
year
month
label
label_binary
expected_gt_px
mask_gt_px
name_mask_match
original_difficulty
group_date
group_month
group_year
group_id
manual_decision
manual_confidence
image_quality
scene_tags
has_cloud
has_thin_cloud
has_wake
has_ship
has_coast
has_sunglint
has_dark_water
has_turbid_water
has_possible_debris
notes
```

---

## 04 — `04_annotate_patch_quality.py`

Renombrar desde:

```text
13_annotate_patches_quality.py
```

Objetivo:

- revisar patches visualmente;
- completar etiquetas manuales;
- guardar metadata anotada.

Salida:

```text
outputs/04_annotate_patch_quality/dataset_metadata_annotated.csv
```

Si modifica el metadata base, crear backup antes.

---

## 05 — `05_run_unet_marida.py`

Renombrar desde:

```text
02_predict_unet.py
```

Salida:

```text
outputs/05_run_unet_marida/
summary_unet.csv
masks/
```

---

## 06 — `06_run_rf_marida.py`

Renombrar desde:

```text
03_predict_rf.py
```

Salida:

```text
outputs/06_run_rf_marida/
summary_rf.csv
masks/
```

---

## 07 — `07_run_resnet_marida.py`

Renombrar desde:

```text
04_predict_resnet.py
```

Salida:

```text
outputs/07_run_resnet_marida/
summary_resnet.csv
predictions/
```

---

## 08 — `08_run_spectral_indices.py`

Renombrar desde:

```text
05_predict_indices.py
```

Debe calcular:

```text
FDI
NDVI
FDI+NDVI
```

Salida:

```text
outputs/08_run_spectral_indices/
summary_indices.csv
masks/
```

---

## 09 — `09_unify_predictions.py`

Renombrar desde:

```text
06_unify_predictions.py
```

Objetivo:

- unificar predicciones de U-Net, RF, ResNet e índices.

Salida:

```text
outputs/09_unify_predictions/predictions_master.csv
```

---

## 10 — `10_evaluate_pretrained_baselines.py`

Fusionar:

```text
09_evaluate.py
14_consolidate_current_results.py
```

Objetivo:

Evaluar modelos que no se entrenan con el dataset:

```text
U-Net MARIDA
RF MARIDA
ResNet MARIDA
FDI
NDVI
FDI+NDVI
```

Salidas:

```text
outputs/10_evaluate_pretrained_baselines/patch_level_metrics.csv
outputs/10_evaluate_pretrained_baselines/pixel_level_segmentation_metrics.csv
outputs/10_evaluate_pretrained_baselines/false_positives_by_difficulty.csv
outputs/10_evaluate_pretrained_baselines/segmentation_noise_on_negatives.csv
outputs/10_evaluate_pretrained_baselines/baseline_summary.md
```

Importante:

- no usar GroupKFold aquí;
- sí incluir análisis por fecha/calidad si hay metadata suficiente;
- separar patch-level y pixel-level.

---

## 11 — `11_error_analysis_baselines.py`

Fusionar:

```text
10_error_analysis.py
15_error_analysis_v2.py
```

Objetivo:

- generar ejemplos visuales de FP/FN;
- distinguir total de errores y ejemplos exportados.

Salidas:

```text
outputs/11_error_analysis_baselines/error_examples_summary.csv
outputs/11_error_analysis_baselines/examples/
```

Columnas obligatorias:

```text
method
total_fp
total_fn
exported_fp_examples
exported_fn_examples
export_dir
```

---

## 12 — `12_analysis_by_date.py`

Mantener desde:

```text
16_analysis_by_date.py
```

Objetivo:

- distribución temporal;
- métricas por fecha;
- fechas dominantes.

Salidas:

```text
outputs/12_analysis_by_date/date_distribution.csv
outputs/12_analysis_by_date/metrics_by_date.csv
outputs/12_analysis_by_date/top_dates_summary.csv
```

---

## 13 — `13_build_grouped_splits.py`

Renombrar desde:

```text
18_build_grouped_splits.py
```

Objetivo:

- crear folds agrupados por fecha para modelos propios.

Salidas:

```text
outputs/13_build_grouped_splits/folds.csv
outputs/13_build_grouped_splits/split_summary.md
```

Requisitos:

- intentar `StratifiedGroupKFold` si está disponible;
- si no, usar `GroupKFold`;
- advertir si los folds quedan desbalanceados;
- no partir la misma fecha entre train y test.

---

## 14 — `14_run_xgboost_experiments.py`

Nuevo script unificado.

Sustituye/fusiona:

```text
08_train_xgboost.py
19_train_xgboost_grouped_cv.py
20_compare_xgboost_exploratory_vs_grouped.py
```

Modos:

```bash
python scripts/14_run_xgboost_experiments.py --mode exploratory
python scripts/14_run_xgboost_experiments.py --mode grouped
python scripts/14_run_xgboost_experiments.py --mode compare
python scripts/14_run_xgboost_experiments.py --mode all
```

Salidas:

```text
outputs/14_run_xgboost_experiments/exploratory/
outputs/14_run_xgboost_experiments/grouped/
outputs/14_run_xgboost_experiments/comparison/
```

Ver sección de XGBoost arriba.

---

## 15 — `15_compute_sam_features.py`

Renombrar desde:

```text
24_compute_sam_features.py
```

Salida:

```text
outputs/15_compute_sam_features/sam_features.csv
```

SAM debe tratarse como score/feature de similitud espectral, no como confirmación de plástico.

---

## 16 — `16_compute_glcm_features.py`

Renombrar desde:

```text
25_compute_glcm_features.py
```

Salida:

```text
outputs/16_compute_glcm_features/glcm_features.csv
```

---

## 17 — `17_build_feature_table_v2.py`

Renombrar desde:

```text
26_build_feature_table_v2.py
```

Objetivo:

Unir:

```text
metadata
predicciones de baselines
features espectrales
SAM
GLCM
calidad manual
predicciones externas si existen
```

Salida:

```text
outputs/17_build_feature_table_v2/feature_table_v2.csv
```

---

## 18 — `18_train_classical_models_grouped.py`

Renombrar desde:

```text
23_train_classical_models_grouped_cv.py
```

Objetivo:

Entrenar modelos propios con validación agrupada:

```text
Logistic Regression
Random Forest
HistGradientBoosting
XGBoost
```

Salidas:

```text
outputs/18_train_classical_models_grouped/fold_metrics.csv
outputs/18_train_classical_models_grouped/out_of_fold_predictions.csv
outputs/18_train_classical_models_grouped/summary_metrics.csv
outputs/18_train_classical_models_grouped/classical_models_summary.md
```

---

## 19 — `19_run_ablation_grouped.py`

Renombrar desde:

```text
27_ablation_grouped_cv.py
```

Objetivo:

- ablación por familias de features;
- validación agrupada por fecha.

Salida:

```text
outputs/19_run_ablation_grouped/ablation_results.csv
outputs/19_run_ablation_grouped/ablation_summary.md
```

---

## 20 — `20_run_external_models.py`

No olvidar este script.

Renombrar desde:

```text
22_run_external_model.py
```

Objetivo:

- ejecutar modelos externos/preentrenados de la literatura;
- actuar como wrapper;
- convertir salidas al formato común.

Ejemplos de modelos externos posibles:

```text
marinedebrisdetector
MADOS / MariNeXt
POS2IDON
SADMA / ResAttUNet
```

Uso esperado:

```bash
python scripts/20_run_external_models.py --model marinedebrisdetector
```

Salidas:

```text
outputs/20_run_external_models/<model_name>/
├── predictions.csv
├── masks/
└── run_summary.md
```

El script debe manejar errores de dependencias externas sin romper toda la pipeline. Si un modelo no está instalado, generar `run_summary.md` con el error y continuar.

---

## 21 — `21_evaluate_external_models.py`

No olvidar este script.

Mantener/renombrar desde:

```text
21_evaluate_external_models.py
```

Objetivo:

- evaluar predicciones de modelos externos;
- convertirlas al formato común de métricas;
- compararlas con baselines si procede.

Salidas:

```text
outputs/21_evaluate_external_models/
├── external_model_patch_metrics.csv
├── external_model_pixel_metrics.csv
├── external_models_by_date.csv
└── external_models_summary.md
```

Importante:

- si el modelo externo es preentrenado, no usar GroupKFold;
- evaluar globalmente y por fecha/calidad;
- si produce máscaras, calcular métricas pixel-level;
- si solo produce score patch-level, calcular métricas patch-level.

---

## 22 — `22_visualize_predictions.py`

Nuevo visualizador final.

Debe basarse en el antiguo:

```text
11_visualizacion.py
```

pero estar al final de la pipeline.

## Modos

### Modo original

```bash
python scripts/22_visualize_predictions.py --mode original
```

Alias:

```bash
python scripts/22_visualize_predictions.py --mode 1
```

Debe comportarse igual que el antiguo `11_visualizacion.py`, salvo por rutas nuevas.

Debe permitir:

- ir pasando patch por patch;
- ver las mismas máscaras/modelos que antes;
- conservar la misma lógica de navegación;
- mostrar siempre RGB + pseudo-máscara + predicciones.

Modelos habituales del modo original:

```text
RGB
pseudo-GT / máscara Nature-derived
U-Net MARIDA
RF MARIDA
FDI
NDVI
FDI+NDVI
```

Si el original mostraba otros, conservarlos.

### Modo extended

```bash
python scripts/22_visualize_predictions.py --mode extended
```

Alias:

```bash
python scripts/22_visualize_predictions.py --mode 2
```

Debe añadir modelos no comparados en el visualizador original:

```text
XGBoost exploratory
XGBoost grouped
Logistic Regression grouped
Random Forest grouped
HistGradientBoosting grouped
mejor modelo clásico grouped
SAM score
GLCM features/score si existe
modelo externo si existe
```

Como muchos son patch-level y no tienen máscara, mostrar:

```text
probabilidad / score
label predicha
threshold
fold si es out-of-fold
```

Puede dividir la vista en:

```text
vista A: segmentación
vista B: scores patch-level
```

pero debe permitir seguir navegando patch por patch.

## Navegación mínima

Implementar o conservar:

```text
n / flecha derecha  -> siguiente patch
p / flecha izquierda -> patch anterior
q / esc -> salir
s -> guardar figura actual
```

Argumentos mínimos:

```bash
--mode
--patch
--max-patches
```

Argumentos opcionales:

```bash
--only SI
--only NO
--errors UNet
--difficulty DIFICIL
```

## Outputs

```text
outputs/22_visualize_predictions/
├── saved_original/
├── saved_extended/
└── visualization_log.csv
```

`visualization_log.csv` debe registrar figuras guardadas:

```text
timestamp
mode
patch
output_png
```

Mover el antiguo `11_visualizacion.py` a:

```text
archive/deprecated_scripts/11_visualizacion.py
```

---

## 23 — `23_export_reports.py`

Crear este script.

Objetivo:

Regenerar automáticamente los reportes finales a partir de `outputs/`.

Debe crear/actualizar:

```text
reports/
├── tables/
├── figures/
├── visual_examples/
├── methodology/
├── pipeline_summary.md
└── final_results_index.md
```

---

# Qué debe copiar `23_export_reports.py`

## A `reports/tables/`

Copiar si existen:

```text
outputs/10_evaluate_pretrained_baselines/patch_level_metrics.csv
outputs/10_evaluate_pretrained_baselines/pixel_level_segmentation_metrics.csv
outputs/10_evaluate_pretrained_baselines/false_positives_by_difficulty.csv
outputs/10_evaluate_pretrained_baselines/segmentation_noise_on_negatives.csv

outputs/12_analysis_by_date/date_distribution.csv
outputs/12_analysis_by_date/metrics_by_date.csv
outputs/12_analysis_by_date/top_dates_summary.csv

outputs/14_run_xgboost_experiments/exploratory/metrics.csv
outputs/14_run_xgboost_experiments/grouped/summary_metrics.csv
outputs/14_run_xgboost_experiments/grouped/fold_metrics.csv
outputs/14_run_xgboost_experiments/comparison/xgboost_comparison.csv

outputs/18_train_classical_models_grouped/summary_metrics.csv
outputs/18_train_classical_models_grouped/fold_metrics.csv

outputs/19_run_ablation_grouped/ablation_results.csv

outputs/21_evaluate_external_models/external_model_patch_metrics.csv
outputs/21_evaluate_external_models/external_model_pixel_metrics.csv
```

Renombrar con prefijos para evitar colisiones:

```text
10_patch_level_metrics.csv
10_pixel_level_segmentation_metrics.csv
12_date_distribution.csv
14_xgboost_exploratory_metrics.csv
14_xgboost_grouped_summary_metrics.csv
14_xgboost_comparison.csv
18_classical_models_summary_metrics.csv
19_ablation_results.csv
21_external_model_patch_metrics.csv
```

---

## A `reports/figures/`

Copiar `.png`, `.jpg`, `.jpeg`, `.pdf` relevantes desde:

```text
outputs/10_evaluate_pretrained_baselines/
outputs/11_error_analysis_baselines/
outputs/12_analysis_by_date/
outputs/14_run_xgboost_experiments/
outputs/18_train_classical_models_grouped/
outputs/19_run_ablation_grouped/
outputs/21_evaluate_external_models/
outputs/22_visualize_predictions/saved_original/
outputs/22_visualize_predictions/saved_extended/
```

Organizar en subcarpetas:

```text
reports/figures/baselines/
reports/figures/errors/
reports/figures/xgboost/
reports/figures/classical_models/
reports/figures/ablation/
reports/figures/external_models/
reports/figures/visualization/
```

---

## A `reports/visual_examples/`

Copiar ejemplos desde:

```text
outputs/22_visualize_predictions/saved_original/
outputs/22_visualize_predictions/saved_extended/
outputs/11_error_analysis_baselines/examples/
```

Conservar el nombre del patch en el nombre del archivo.

---

# `reports/final_results_index.md`

Generar índice automático:

```markdown
# Final Results Index

## Tables

- `reports/tables/...`

## Figures

- `reports/figures/...`

## Visual examples

- `reports/visual_examples/...`

## Methodology

- `reports/methodology/...`
```

---

# `reports/pipeline_summary.md`

Generar resumen con:

```markdown
# Pipeline Summary

## Dataset

- Nº de patches
- SI / NO
- Fechas únicas
- Fechas más representadas

## Baseline models

- U-Net MARIDA
- RF MARIDA
- ResNet MARIDA
- FDI
- NDVI
- FDI+NDVI

## XGBoost

- Resultado exploratorio
- Resultado grouped-by-date
- Diferencia e interpretación sobre leakage

## Classical models

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

## External models

- Modelos probados
- Estado de ejecución
- Métricas si existen

## Ablation

- Mejor familia de features
- Efecto de SAM
- Efecto de GLCM

## Limitations

- Weak ground truth
- Pseudo-masks
- Temporal grouping
- Pixel-level limitations
```

---

# Archivar scripts antiguos

Mover a `archive/deprecated_scripts/` si existen:

```text
00_explore_candidates.py
01_download_dataset.py
02_predict_unet.py
03_predict_rf.py
04_predict_resnet.py
05_predict_indices.py
06_unify_predictions.py
07_build_xgboost_dataset.py
08_train_xgboost.py
09_evaluate.py
10_error_analysis.py
11_visualizacion.py
12_build_dataset_metadata.py
13_annotate_patches_quality.py
14_consolidate_current_results.py
15_error_analysis_v2.py
16_analysis_by_date.py
17_build_group_ids.py
18_build_grouped_splits.py
19_train_xgboost_grouped_cv.py
20_compare_xgboost_exploratory_vs_grouped.py
21_evaluate_external_models.py
22_run_external_model.py
23_train_classical_models_grouped_cv.py
24_compute_sam_features.py
25_compute_glcm_features.py
26_build_feature_table_v2.py
27_ablation_grouped_cv.py
```

Pero solo después de crear sus equivalentes nuevos.

---

# Refactorización de imports

Todos los scripts deben empezar con bloque de bootstrap:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Y luego imports desde `src`.

Ejemplos:

```python
from src.common.config import OUTPUTS_DIR, REPORTS_DIR, DATA_DIR
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.grouped_validation import make_grouped_splits
from src.visualization.overlays import plot_rgb_with_mask
```

No dejar imports antiguos.

---

# Config central

Crear o corregir:

```text
src/common/config.py
```

Debe contener:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

DATA_DIR = PROJECT_ROOT / "data"
PATCHES_DIR = DATA_DIR / "processed" / "patches"
MASKS_DIR = DATA_DIR / "processed" / "masks"

METADATA_DIR = OUTPUTS_DIR / "03_build_dataset_metadata"
METADATA_CSV = METADATA_DIR / "dataset_metadata.csv"
METADATA_GROUPED_CSV = METADATA_DIR / "dataset_metadata_with_groups.csv"

PREDICTIONS_MASTER = OUTPUTS_DIR / "09_unify_predictions" / "predictions_master.csv"
FEATURE_TABLE_V2 = OUTPUTS_DIR / "17_build_feature_table_v2" / "feature_table_v2.csv"
```

Si los patches actuales están en otra ruta, adaptar y documentar.

---

# README nuevo

Crear o reescribir `README.md` con:

```markdown
# TFG Marine Plastic Detection

## 1. Objetivo

## 2. Estructura del proyecto

## 3. Orden de ejecución

## 4. Dataset y weak ground truth

## 5. Modelos evaluados

## 6. Validación

## 7. Visualización

## 8. Reportes

## 9. Resultados principales

## 10. Limitaciones
```

Incluir orden de ejecución recomendado:

```bash
python scripts/00_check_environment.py

python scripts/03_build_dataset_metadata.py
python scripts/10_evaluate_pretrained_baselines.py
python scripts/11_error_analysis_baselines.py
python scripts/12_analysis_by_date.py
python scripts/13_build_grouped_splits.py

python scripts/14_run_xgboost_experiments.py --mode all

python scripts/15_compute_sam_features.py
python scripts/16_compute_glcm_features.py
python scripts/17_build_feature_table_v2.py
python scripts/18_train_classical_models_grouped.py
python scripts/19_run_ablation_grouped.py

python scripts/20_run_external_models.py --model marinedebrisdetector
python scripts/21_evaluate_external_models.py

python scripts/22_visualize_predictions.py --mode original
python scripts/22_visualize_predictions.py --mode extended

python scripts/23_export_reports.py
```

---

# Advertencias metodológicas obligatorias

Incluir en README, summaries y pipeline_summary:

```text
Las máscaras usadas en este proyecto son pseudo-máscaras derivadas del catálogo Nature/litter windrows. 
No constituyen ground truth pixel-perfect ni confirmación material directa de plástico. 
Por ello, las métricas pixel-level deben interpretarse como evaluación de localización aproximada frente a weak labels, no como validación absoluta de plástico puro.
```

Incluir también:

```text
La validación agrupada por fecha solo se aplica a modelos entrenados con el dataset propio, como XGBoost o modelos clásicos. 
Los modelos MARIDA preentrenados y los índices espectrales se evalúan globalmente y por fecha/calidad, pero no requieren GroupKFold porque no se entrenan con este dataset.
```

Para XGBoost:

```text
El XGBoost exploratorio se mantiene únicamente como referencia comparativa. 
El resultado grouped-by-date es el que debe emplearse como resultado principal.
```

---

# Reporte de refactorización

Generar:

```text
reports/refactor_report.md
```

Debe incluir:

```markdown
# Refactor Report

## Scripts renombrados

## Scripts fusionados

## Scripts archivados

## Outputs movidos

## Imports corregidos

## Rutas corregidas

## Nuevos scripts creados

## Pendientes manuales
```

---

# Comprobaciones automáticas

Después de refactorizar, ejecutar:

```bash
python -m py_compile scripts/*.py
```

Si PowerShell da problemas con el comodín:

```powershell
Get-ChildItem scripts\*.py | ForEach-Object { python -m py_compile $_.FullName }
```

También comprobar:

```bash
python scripts/00_check_environment.py
```

---

# Criterios de aceptación

La refactorización se considera correcta si:

1. Existe una pipeline numerada de `00` a `23`.
2. No quedan scripts ejecutables dentro de `src/`.
3. `src/` contiene solo módulos reutilizables.
4. Los imports antiguos han sido sustituidos por imports desde `src`.
5. No quedan rutas absolutas tipo `C:\...`.
6. Existe `14_run_xgboost_experiments.py` con modos:
   - `exploratory`
   - `grouped`
   - `compare`
   - `all`
7. XGBoost exploratorio y grouped se guardan en subcarpetas separadas.
8. Existe `20_run_external_models.py`.
9. Existe `21_evaluate_external_models.py`.
10. Existe `22_visualize_predictions.py`.
11. `22_visualize_predictions.py --mode original` replica el comportamiento del antiguo `11_visualizacion.py`.
12. `22_visualize_predictions.py --mode extended` añade modelos nuevos y permite navegar patch por patch.
13. En ambos modos se ve RGB + pseudo-máscara + predicciones/scores.
14. Existe `23_export_reports.py`.
15. `23_export_reports.py` genera `reports/tables/`, `reports/figures/`, `reports/visual_examples/`, `pipeline_summary.md` y `final_results_index.md`.
16. Los scripts antiguos se archivan, no se eliminan.
17. `python -m py_compile scripts/*.py` no da errores.
18. Existe `reports/refactor_report.md`.
19. README actualizado explica estructura, ejecución, XGBoost exploratorio vs grouped, weak ground truth y limitaciones.

---

# Orden de implementación para Codex

Implementar en este orden:

```text
1. Crear estructura final de carpetas.
2. Crear/corregir src/common/config.py y paths.
3. Renombrar scripts principales.
4. Fusionar metadata + group ids en script 03.
5. Fusionar evaluación baseline en script 10.
6. Fusionar error analysis en script 11.
7. Crear grouped splits en script 13.
8. Crear script único de XGBoost en script 14 con modos.
9. Renombrar SAM, GLCM, feature table, modelos clásicos y ablation.
10. Mantener/crear scripts 20 y 21 para modelos externos.
11. Crear visualizador final 22 con modo original y extended.
12. Crear exportador de reportes 23.
13. Archivar scripts antiguos.
14. Corregir imports y rutas.
15. Crear README.
16. Crear refactor_report.md.
17. Ejecutar py_compile.
18. No cambiar resultados científicos salvo que sea necesario por rutas/imports.
```
