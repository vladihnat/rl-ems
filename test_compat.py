# test_compat_v4.py
import numpy as np
# np.product was removed in NumPy 2.0; pymgrid 1.4.1 still uses it internally
if not hasattr(np, 'product'):
    np.product = np.prod

import pymgrid
from pymgrid import Microgrid
from pymgrid.modules import (BatteryModule, RenewableModule, GridModule, LoadModule)
import gymnasium as gym
import numpy as np

print(f"pymgrid version: {pymgrid.__version__}")
print(f"gymnasium version: {gym.__version__}")

# --- 1. Créer un microgrid minimal ---
# Pour cela on construit les 4 modules de base : batterie, source renouvelable, charge, et grid
# On definit ce test pour une journee de 24 heures
# Batteries 
small_battery = BatteryModule(min_capacity=10,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=0.9,
                              init_soc=0.2)

large_battery = BatteryModule(min_capacity=10,
                              max_capacity=1000,
                              max_charge=10,
                              max_discharge=10,
                              efficiency=0.7,
                              init_soc=0.2)

# Load and PV modules : 
load_ts = 100 + 100 * np.random.rand(24) # random load data in the range [100, 200].
pv_ts   = 200 * np.random.rand(24) # random pv data in the range [0, 200].

load = LoadModule(time_series=load_ts)
pv   = RenewableModule(time_series=pv_ts)

# GridModule nécessite un time_series pour les prix import/export et CO2 prod x kWh 
grid_ts = [0.2, 0.1, 0.5] * np.ones((24, 3)) # [prix_import, prix_export, co2_prod] sur 24h
grid = GridModule(max_import=100, max_export=100, time_series=grid_ts)

modules = [
    small_battery,
    large_battery,
    ('pv', pv),
    load,
    grid
]

env = Microgrid(modules)
print("\n✅ Microgrid créé avec succès")
print(env)


'''
NOTE : 

Pour acceder aux modules dans la microgrid on peut le faire par name ou par key : 
print(microgrid.modules.pv)
print(microgrid.modules.grid is microgrid.modules['grid'])
'''


# --- 2. Tester reset() ---
print("\n[TEST] reset()...")
env.reset()
obs = env.state_series().to_frame()
print(f"  Type retourné: {type(obs)}")
print(f"  ✅ reset() OK")
print(obs)

# --- 3. Tester step() ---
print("\n[TEST] step() avec action structurée...")

# Lire l'état courant
load_val = -1.0 * env.modules.load.item().current_load      # négatif = consommation
pv_val   =        env.modules.pv.item().current_renewable   # positif = production
net_load = load_val + pv_val  # positif = surplus PV, négatif = déficit

print(f"  load    : {load_val:.2f} kWh")
print(f"  pv      : {pv_val:.2f}  kWh")
print(f"  net_load: {net_load:.2f} kWh")

# Stratégie simple : on décharge les batteries si déficit, sinon on n'agit pas
if net_load < 0:
    # Déficit : on essaie de couvrir avec battery[0] puis battery[1]
    deficit = -net_load  # valeur positive

    bat0_action = min(deficit, small_battery.max_production)
    deficit    -= bat0_action

    bat1_action = min(deficit, large_battery.max_production)
    deficit    -= bat1_action

    # Le reste est importé du grid
    grid_import = deficit
else:
    # Surplus PV : on ne décharge pas les batteries
    bat0_action = 0.0
    bat1_action = 0.0
    grid_import = 0.0

control = {
    "battery": [bat0_action, bat1_action],
    "grid":    [grid_import]
}

print(f"\n  Action envoyée: {control}")

obs, reward, done, info = env.step(control, normalized=False)
print(f"  ✅ step() OK — 4 valeurs (obs, reward, done, info)")

print(f"  reward : {reward:.4f}")
print(f"  info   : {info}")

# --- 4. Tester check_env() SB3 ---
# NOTE : check_env() attend l'API gymnasium standard (step/reset/action_space).
# pymgrid.Microgrid natif n'expose PAS action_space ni step() → check_env() échouera.
# Il faudra construire un wrapper gymnasium custom autour de Microgrid pour SB3.
# Ce wrapper sera l'étape suivante du projet.
# print("\n[INFO] check_env() SB3 non testé ici : pymgrid nécessite un wrapper")
# print("       gymnasium custom pour exposer action_space et step() à SB3.")
# print("       → Prochaine étape : implémenter MicrogridEnv(gym.Env)")