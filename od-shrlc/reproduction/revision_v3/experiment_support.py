"""Benchmark signals, reproducible measurement perturbations, and nested ARX models."""
import numpy as np


class MeasurementStream:
    def __init__(self, steps, seed, sigma=0.0, bias=0.0):
        self.sigma=float(sigma)
        self.bias=float(bias)
        if self.sigma<0:
            raise ValueError('Noise sigma must be nonnegative')
        # Independent of PyTorch/replay random number consumption; indexed by step.
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),827361]))
        self.errors=rng.normal(0,self.sigma,int(steps)+1)+self.bias

    def observe(self, measurement, index):
        observed=dict(measurement)
        observed['reaTZon_y']=float(measurement['reaTZon_y'])+float(self.errors[index])
        return observed


def forecast_features(payload, price_name, tout_name, solar_name, expected_time, step=900):
    required=[price_name,tout_name,solar_name,'LowerSetp[1]','UpperSetp[1]','time']
    if any(n not in payload for n in required):
        raise RuntimeError('Required benchmark forecast signal missing')
    arrays={n:np.asarray(payload[n],dtype=float) for n in required}
    if abs(arrays['time'][0]-expected_time)>1e-3:
        raise RuntimeError('Forecast does not start at current simulator time')
    if any(len(x)<25 or not np.isfinite(x).all() for x in arrays.values()):
        raise RuntimeError('Non-finite or truncated 6-hour forecast')
    if not np.allclose(np.diff(arrays['time'][:25]),step):
        raise RuntimeError('Unexpected forecast interval')
    lo=arrays['LowerSetp[1]'][:24]-273.15
    hi=arrays['UpperSetp[1]'][:24]-273.15
    if not np.all(lo<hi):
        raise RuntimeError('Inconsistent comfort limits')
    tout=arrays[tout_name][:24]-273.15
    solar=arrays[solar_name][:24]
    price=arrays[price_name][:24]
    f=dict(Tout_mean_6h=float(tout.mean()),Tout_max_6h=float(tout.max()),
           Qsol_mean_6h=float(solar.mean()),Qsol_max_6h=float(solar.max()),
           Qsol_sum_6h=float(solar.sum()*step/3600),
           price_mean_6h=float(price.mean()),price_max_6h=float(price.max()),
           price_min_6h=float(price.min()),Tlow_mean_6h=float(lo.mean()),
           Tlow_max_6h=float(lo.max()),Thigh_mean_6h=float(hi.mean()),
           Thigh_min_6h=float(hi.min()),current_low_C=float(lo[0]),current_high_C=float(hi[0]))
    return f,float(price[0]),float(lo[0]),float(hi[0])


class OnlineARXPredictor:
    """Order 1 reproduces original 5 coefficients; order 2 adds dT and delayed u.

    No historical trajectory is supplied. The first lagged-temperature difference
    and previous action start at zero. Same RLS forgetting/projection conventions.
    This is an ARX lag-order ablation, not a physical 2R2C identification.
    """
    def __init__(self, order=1, forgetting=.998):
        if order not in (1,2):raise ValueError('order must be 1 or 2')
        self.order=order
        self.theta=np.array([.95,.03,.05,.30,0]+([0.,0.] if order==2 else []))
        self.P=np.eye(len(self.theta))*100.
        self.forgetting=float(forgetting)
        self.last_temp=None
        self.last_u=0.
        self.lower=np.array([.70,-.20,-2.,0.,-10.]+([-1.,0.] if order==2 else []))
        self.upper=np.array([1.05,.20,2.,3.,10.]+([1.,3.] if order==2 else []))

    def make_phi(self,Tz_C,Tout_C,Qsol_Wm2,u_hp):
        v=[float(Tz_C),float(Tout_C),float(Qsol_Wm2)/1000,float(u_hp),1.]
        if self.order==2:
            v += [0. if self.last_temp is None else float(Tz_C)-self.last_temp,self.last_u]
        return np.array(v)

    def predict(self,phi):return float(self.theta@phi)

    def predict_actions(self,Tz_C,Tout_C,Qsol_Wm2,actions):
        phi=self.make_phi(Tz_C,Tout_C,Qsol_Wm2,0.)
        return float(self.theta@phi)+self.theta[3]*np.asarray(actions,dtype=float)

    def update(self,phi_prev,Tz_next_C):
        phi=phi_prev.reshape(-1,1)
        theta=self.theta.reshape(-1,1)
        denom=self.forgetting+(phi.T@self.P@phi).item()
        gain=(self.P@phi)/denom
        pred=(theta.T@phi).item()
        error=float(Tz_next_C)-pred
        theta=theta+gain*error
        self.P=(self.P-gain@phi.T@self.P)/self.forgetting
        self.theta=np.clip(theta.flatten(),self.lower,self.upper)
        if not np.isfinite(self.theta).all() or not np.isfinite(self.P).all():
            raise RuntimeError('Non-finite RLS state')
        return pred,error,self.theta.copy()

    def commit(self,temperature,action):
        self.last_temp=float(temperature)
        self.last_u=float(action)
