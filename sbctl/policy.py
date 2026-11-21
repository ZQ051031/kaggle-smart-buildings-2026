"""NumPy-only controller; all physical temperatures are Kelvin."""
import json
from pathlib import Path
import re

import numpy as np

DEFAULT_PARAMS = dict(kp=0.07, ki=0.0035, integral_limit=1.0,
    prestart_hours=2.0, cooling_hold_steps=288.0, heating_setback=2.0,
    damper_gain=1.0, reheat_gain=1.0, fan_base=0.1, fan_gain=0.7,
    sat_base=295.0, sat_gain=10.0, water_base=0.0, water_gain=1.0,
    on_threshold=0.03, econ_advantage=2.0)
PARAM_BOUNDS = dict(kp=(0.01,0.3), ki=(0.0001,0.02), integral_limit=(0.2,2.0),
    prestart_hours=(0.0,4.0), cooling_hold_steps=(0.0,576.0), heating_setback=(0.0,5.0),
    damper_gain=(0.2,3.0), reheat_gain=(0.2,3.0), fan_base=(0.1,0.5), fan_gain=(0.0,1.0),
    sat_base=(290.0,300.0), sat_gain=(0.0,20.0), water_base=(0.0,1.0),
    water_gain=(0.0,2.0), on_threshold=(0.0,0.3), econ_advantage=(0.0,10.0))


class TunedTransferPolicy:
    name = 'tuned'

    def __init__(self, building_meta_data, action_spec=None, params=None):
        m = self.metadata = building_meta_data
        self.params = DEFAULT_PARAMS.copy()
        path = Path(__file__).with_name('params.json')
        if path.exists():
            self.params.update(json.loads(path.read_text()))
        self.params.update(params or {})
        for key, value in self.params.items():
            if key not in PARAM_BOUNDS or not np.isfinite(value):
                raise ValueError(f'Invalid parameter: {key}')
        self.nc = int(m['n_continuous_action'])
        self.nd = int(m['n_discrete_action'])
        self.ti = np.array(m.get('temp_indices', []), dtype=int)
        self.pi = np.array(m.get('pressure_indices', []), dtype=int)
        self.ri = np.array(m.get('recirculation_indices', []), dtype=int)
        self.wi = np.array(m.get('water_indices', []), dtype=int)
        self.k = len(self.ti)
        names = m.get('observation_field_names', [])
        self.fields = {n:i for i,n in enumerate(names)}
        self.zones = list(m.get('zone_action_indices', {}))
        self.zi = np.array([self.fields.get(z+'_zone_air_temperature_sensor', -1) for z in self.zones])
        self.loi = np.array([self.fields.get(z+'_zone_air_heating_temperature_setpoint_sensor', -1) for z in self.zones])
        self.hii = np.array([self.fields.get(z+'_zone_air_cooling_temperature_setpoint_sensor', -1) for z in self.zones])
        self.di = np.array([m['zone_action_indices'][z].get('supply_air_damper_percentage_command', -1) for z in self.zones], dtype=int)
        self.vi = np.array([m['zone_action_indices'][z].get('reheat_valve_percentage_command', -1) for z in self.zones], dtype=int)
        assignments = m.get('ahu_zone_assignments', {})
        ahus = sorted(assignments, key=lambda s:int(re.search(r'\d+',s)[0]) if re.search(r'\d+',s) else 0)
        self.groups = [np.array([j for j,z in enumerate(self.zones) if z in [r.replace('room_', 'VAV_') for r in assignments[a]]], dtype=int) for a in ahus]
        self.oati = [self.fields.get('ahs_'+a+'_outside_air_temperature_sensor', -1) for a in ahus]
        norm = m.get('observation_normalization_metadata', {}).get('zone_air_temperature_sensor', {})
        self.mean = norm.get('sample_mean', 295.0)
        self.std = norm.get('sample_std', 5.0)
        self.integral = np.zeros(len(self.zones))
        self.hold = np.zeros(self.k)

    @staticmethod
    def _read(obs, indices, default):
        indices = np.asarray(indices, dtype=int)
        result = np.full(indices.shape, default, dtype=float)
        valid = (indices >= 0) & (indices < len(obs))
        result[valid] = obs[indices[valid]]
        return np.where(np.isfinite(result), result, default)

    def get_action(self, obs):
        obs = np.asarray(obs).reshape(-1)
        p = self.params
        c = np.full(self.nc, -1.0, dtype=np.float32)
        d = np.zeros(self.nd, dtype=np.int32)
        def field(name, default):
            return self._read(obs, [self.fields.get(name,-1)], default)[0]
        hour = (np.arctan2(field('hod_sin_000',0),field('hod_cos_000',1)) % (2*np.pi))*24/(2*np.pi)
        day = int(round((np.arctan2(field('dow_sin_000',0),field('dow_cos_000',1)) % (2*np.pi))*7/(2*np.pi))) % 7
        morning = self.metadata.get('morning_start_hour',8)
        evening = self.metadata.get('evening_start_hour',18)
        occupied = field('comfort_mode_now', float(day<5 and morning<=hour<evening)) > 0.5
        soon = field('comfort_mode_soon',0) > 0.5
        # Sensor bands and comfort flags are authoritative for holidays/time zones.
        scheduled = occupied or soon or (day<5 and morning-p['prestart_hours']<=hour<morning)
        low, high = self.metadata.get('comfort_temp_window',(294.0,297.0))
        lo = self._read(obs,self.loi,low)
        hi = self._read(obs,self.hii,high)
        valid = hi>lo
        lo = np.where(valid,lo,low)
        hi = np.where(valid,hi,high)
        lo = np.maximum(lo,low-p['heating_setback'])
        hi = np.minimum(hi,high)
        if scheduled:
            lo = np.full_like(lo,low)
        target = (lo+hi)/2
        temp = self._read(obs,self.zi,(np.mean(target)-self.mean)/self.std if len(target) else 0)*self.std+self.mean
        error = target-temp
        trial = np.clip(self.integral+p['ki']*error,-p['integral_limit'],p['integral_limit'])
        raw = p['kp']*error+trial
        self.integral = np.where((np.abs(raw)<=1) | (error*raw<0),trial,self.integral)
        u = np.clip(p['kp']*error+self.integral,-1,1)
        damper = np.clip(np.abs(u)*p['damper_gain'],0,1)
        reheat = np.clip(u*p['reheat_gain'],0,1)
        for idx,val in ((self.di,damper),(self.vi,reheat)):
            mask = idx>=0
            c[idx[mask]] = 2*val[mask]-1
        for a in range(self.k):
            group = self.groups[a] if a<len(self.groups) else np.array([],dtype=int)
            demand = float(np.mean(u[group])) if len(group) else 0.0
            effort = float(np.max(np.abs(u[group]))) if len(group) else 0.0
            if demand < -p['on_threshold']:
                self.hold[a] = p['cooling_hold_steps']
            active = scheduled or effort>p['on_threshold'] or self.hold[a]>0
            self.hold[a] = max(0,self.hold[a]-1)
            if a<self.nd-1:
                d[a] = active
            sat = np.clip(p['sat_base']+p['sat_gain']*demand,285,305)
            c[self.ti[a]] = (sat-285)/10-1
            if a<len(self.pi):
                c[self.pi[a]] = 2*np.clip(p['fan_base']+p['fan_gain']*effort,0,1)-1
            if a<len(self.ri):
                recirc = 0.7
                outside = self._read(obs,[self.oati[a] if a<len(self.oati) else -1],295)[0]
                inside = float(np.mean(temp[group])) if len(group) else 295.0
                if demand<0 and inside-outside>p['econ_advantage']:
                    recirc = np.clip(1-(inside-sat)/(inside-outside),0,0.7)
                c[self.ri[a]] = 2*recirc-1
        heat = float(np.max(reheat)) if len(reheat) else 0.0
        c[self.wi] = 2*np.clip(p['water_base']+p['water_gain']*heat,0,1)-1
        if len(self.pi)>self.k:
            c[self.pi[self.k:]] = 1 if heat>p['on_threshold'] else -1
        if self.nd:
            d[-1] = heat>p['on_threshold']
        return dict(continuous_action=np.clip(c,-1,1), discrete_action=d)
