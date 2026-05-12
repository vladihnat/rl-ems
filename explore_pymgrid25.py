# explore_pymgrid25_v3.py
from pymgrid import PROJECT_PATH, Microgrid

def inspect_microgrid(mg):
    has_grid    = len(mg.modules.get("grid",    [])) > 0
    has_battery = len(mg.modules.get("battery", [])) > 0
    has_pv      = len(mg.modules.get("pv",      [])) > 0
    has_genset  = len(mg.modules.get("genset",  [])) > 0
    return has_grid, has_battery, has_pv, has_genset

print(f"{'N°':<5} {'Grid?':<8} {'PV?':<8} {'Battery?':<10} {'Genset?'}")
print("-" * 45)

candidates = []

for i in range(25):
    yaml_file = PROJECT_PATH / f'data/scenario/pymgrid25/microgrid_{i}/microgrid_{i}.yaml'
    mg = Microgrid.load(yaml_file.open('r'))
    has_grid, has_battery, has_pv, has_genset = inspect_microgrid(mg)

    print(f"{i:<5} {'✅' if has_grid else '❌':<8} {'✅' if has_pv else '❌':<8} "
          f"{'✅' if has_battery else '❌':<10} {'✅' if has_genset else '❌'}")

    if has_grid and has_battery and has_pv and not has_genset:
        candidates.append(i)

print(f"\n--- ✅ Candidats PV + battery + grid (sans genset) : {candidates} ---")

for i in candidates:
    yaml_file = PROJECT_PATH / f'data/scenario/pymgrid25/microgrid_{i}/microgrid_{i}.yaml'
    mg = Microgrid.load(yaml_file.open('r'))
    print(f"\nMicrogrid {i}:")

    for bat in mg.modules.get("battery", []):
        print(f"  Battery — capacité: [{bat.min_capacity:.0f}, {bat.max_capacity:.0f}] kWh | "
              f"max charge/discharge: {bat.max_charge:.0f}/{bat.max_discharge:.0f} kW | "
              f"η={bat.efficiency}")

    for grid in mg.modules.get("grid", []):
        print(f"  Grid    — import max: {grid.max_import:.0f} kW | "
              f"export max: {grid.max_export:.0f} kW")

    for pv in mg.modules.get("pv", []):
        print(f"  PV      — production max: {pv.time_series.max():.1f} kW | "
              f"moyenne: {pv.time_series.mean():.1f} kW")

    for load in mg.modules.get("load", []):
        print(f"  Load    — conso max: {load.time_series.max():.1f} kW | "
              f"moyenne: {load.time_series.mean():.1f} kW")