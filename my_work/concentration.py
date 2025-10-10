import numpy as np
from numpy import trapz

def compute_inlet_concentration(time,
                                mode,
                                # for mass/vol flow modes
                                mass_flow=None,      # kg/s array or scalar
                                vol_flow=None,       # m^3/s array or scalar
                                rho_co2=None,        # kg/m^3
                                V_ctrl=None,         # m^3 control volume for normalization
                                # for measured-signal mode
                                S_inlet=None,        # raw inlet signal (same length as time)
                                S_brine=None,
                                S_co2_signal=None,
                                phi_inlet=None,      # porosity at inlet if using bulk basis
                                basis='bulk'         # 'bulk' or 'pore'
                               ):
    """
    Return c_in(t) array (same length as time) in units kg/m^3 (bulk) or kg/m^3(pore) depending on basis.
    - mode: 'mass_flow', 'vol_flow', or 'signal'
    - For 'mass_flow': provide mass_flow (kg/s) and V_ctrl (m^3) -> c_in = mass_flow / V_ctrl
    - For 'vol_flow': provide vol_flow (m^3/s) and rho_co2 (kg/m^3) and V_ctrl
    - For 'signal': provide S_inlet, S_brine, S_co2_signal, rho_co2, basis and phi_inlet if bulk basis
    """
    time = np.asarray(time)
    dt = None
    if len(time) > 1:
        dt = time[1] - time[0]

    if mode == 'mass_flow':
        if mass_flow is None or V_ctrl is None:
            raise ValueError("Provide mass_flow and V_ctrl for mode='mass_flow'")
        m = np.asarray(mass_flow)
        # allow scalar mass_flow (constant)
        if m.size == 1:
            c = np.full_like(time, float(m) / float(V_ctrl))
        else:
            c = np.asarray(m) / float(V_ctrl)
        return c

    if mode == 'vol_flow':
        if vol_flow is None or rho_co2 is None or V_ctrl is None:
            raise ValueError("Provide vol_flow, rho_co2, V_ctrl for mode='vol_flow'")
        q = np.asarray(vol_flow)
        m = np.asarray(q) * float(rho_co2)
        if m.size == 1:
            c = np.full_like(time, float(m) / float(V_ctrl))
        else:
            c = np.asarray(m) / float(V_ctrl)
        return c

    if mode == 'signal':
        # convert signal -> saturation -> concentration
        if S_inlet is None or S_brine is None or S_co2_signal is None or rho_co2 is None:
            raise ValueError("Provide S_inlet, S_brine, S_co2_signal, rho_co2 for mode='signal'")
        S_inlet = np.asarray(S_inlet)
        if S_inlet.shape != time.shape:
            raise ValueError("S_inlet must have same length as time")
        # linear mixing approximation:
        Sg = (S_inlet - S_brine) / (S_co2_signal - S_brine)   # saturation (0..1)
        Sg = np.clip(Sg, 0.0, 1.0)
        if basis == 'pore':
            # concentration per pore volume (kg CO2 / m^3 pore)
            c = Sg * float(rho_co2)
        else:
            # bulk basis: need phi_inlet
            if phi_inlet is None:
                raise ValueError("Provide phi_inlet when basis='bulk'")
            c = Sg * float(phi_inlet) * float(rho_co2)
        return c

    raise ValueError("Unknown mode. Use 'mass_flow', 'vol_flow', or 'signal'.")


def signal_to_voxel_concentration(S_voxel, S_brine, S_co2_signal, rho_co2, phi_i=None, basis='bulk'):
    """
    Convert voxel image signal S_voxel(t) to concentration C_i(t).
    - S_voxel: array time series
    - If basis='bulk', phi_i must be provided (or you will compute phi later via AUC)
    - returns C_i(t) matching same basis as compute_inlet_concentration
    """
    Sg = (np.asarray(S_voxel) - S_brine) / (S_co2_signal - S_brine)
    Sg = np.clip(Sg, 0.0, 1.0)
    if basis == 'pore':
        return Sg * float(rho_co2)
    else:
        if phi_i is None:
            # return pore-basis concentration; caller can divide by inlet AUC to find phi
            return Sg * float(rho_co2)   # caller will compute phi = AUC_voxel / AUC_inlet and then multiply by phi
        else:
            return Sg * float(phi_i) * float(rho_co2)


def compute_auc(time, signal):
    """Simple trapezoidal AUC"""
    return trapz(signal, time)


def estimate_phi_from_aucs(time, C_i_t, c_in_t):
    """
    Given voxel concentration time series C_i(t) and inlet concentration c_in(t)
    (both in same units), compute phi_i = AUC_voxel / AUC_inlet.
    """
    auc_voxel = compute_auc(time, C_i_t)
    auc_in = compute_auc(time, c_in_t)
    if auc_in == 0:
        raise ValueError("Inlet AUC is zero; cannot compute phi.")
    return auc_voxel / auc_in, auc_voxel, auc_in


# ---------------------------
# Example usage (synthetic)
# ---------------------------
if __name__ == "__main__":
    # time grid
    t = np.linspace(0, 200, 201)  # seconds

    #inlet specified by volumetric flow of pure CO2 ---
    Q_in = 1e-5          # m^3/s
    rho_co2 = 700.0      # kg/m^3 (example)
    V_ctrl = 1e-6        # m^3 control volume (= voxel bulk vol)
    c_in = compute_inlet_concentration(t, mode='vol_flow',
                                       vol_flow=Q_in, rho_co2=rho_co2, V_ctrl=V_ctrl)

    # synthetic voxel signals (CT-like) ---
    # Suppose S_brine=1000, S_co2_signal=100, voxel reaches 50% CO2 between t=20..80
    S_brine = 1000.0
    S_co2_signal = 100.0
    S_voxel = np.full_like(t, S_brine)
    # simulate a rectangular pulse in that voxel
    S_voxel[(t >= 20) & (t <= 80)] = S_brine + 0.5 * (S_co2_signal - S_brine)

    # Convert voxel signal to concentration (pore-basis first)
    C_pore = signal_to_voxel_concentration(S_voxel, S_brine, S_co2_signal, rho_co2, basis='pore')

    # If inlet c_in computed in 'bulk' basis, ensure both are same basis.
    # In this synthetic example c_in was computed per bulk voxel (because V_ctrl==voxel vol),
    # so convert pore -> bulk by multiplying by phi_i (if you know phi_i). Here we pretend phi=0.1.
    phi_true = 0.10
    C_bulk = C_pore * phi_true

    # Compute AUCs and estimate phi (pretend we didn't know phi)
    # We'll estimate phi_est = AUC_voxel_bulk / AUC_inlet_bulk
    phi_est, auc_v, auc_in = estimate_phi_from_aucs(t, C_bulk, c_in)
    print("AUC voxel (bulk) = {:.3e}, AUC inlet = {:.3e}, estimated phi = {:.3f}".format(auc_v, auc_in, phi_est))
    print("True phi was {:.3f}".format(phi_true))

    import numpy as np

    phi_volume = np.load("phi_volume.npy")
    C_pore_est = np.load("C_pore_est.npy")
    #c_in_local = np.load("c_in_local.npy")

    print(phi_volume.shape)  # 3D volume of phi
    print(C_pore_est.shape)  # 2D array: (n_voxels, time)
    #print(c_in_local.shape)  # 2D array: (n_voxels, time)

