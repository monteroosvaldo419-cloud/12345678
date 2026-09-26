# Hasty-CR: ruta de entrenamiento profesional

## Principio

El objetivo es medir comportamiento de partida, no proxies aislados.

Mantener `tmp/rl/clone_v4_probe_init.pt` intacto como baseline. Las nuevas pruebas
deben usar nombres descriptivos.

## Behavior Cloning

La colección sigue usando la receta V4 por defecto:

```powershell
python -m sim.clone --episodes 400 --epochs 8 --batch 256 --lr 1e-3 --name clone_v4_retest
```

El selector de checkpoint ahora usa `macro_accuracy` y luego `play_accuracy`, en
lugar de dejar que la exactitud global dominada por HOLD elija el archivo.

Para aumentar la diversidad de estados de entrenamiento:

```powershell
python -m sim.clone --episodes 400 --epochs 8 --batch 256 --lr 1e-3 `
  --opponent-mix "brain:0.5,meta:0.25,mirror:0.25" `
  --name clone_diverse_01
```

Este mix es opt-in y no reemplaza V4 automáticamente.

## PPO

La evaluación puede cubrir varias familias sin cambiar la dieta de entrenamiento:

```powershell
python -m sim.train_ppo --resume tmp/rl/clone_v4_probe_init.pt `
  --name ppo_multi_eval_01 `
  --eval-opponents "brain,meta,mirror"
```

El log conserva resultados por oponente y agrega daño de torres además de
victorias/derrotas y coronas. Una caída fuerte contra una familia ya no puede
quedar escondida por el agregado.

## Diagnóstico rápido

Antes de cualquier corrida larga:

```powershell
python scripts/preflight.py
```

Si falta `tmp/gamedata/csv_logic`, el problema es de datos/entorno y no tiene
sentido gastar tiempo en entrenamiento.

## Search

No activar `decision_engine=search` como reemplazo automático del maestro aún.
El search actual es real y testeado estructuralmente, pero el `OpponentModel`
no genera escenarios de respuesta y el predictor no sustituye una simulación de
combate completa. Primero debe medirse contra `legacy` en `shadow`.
