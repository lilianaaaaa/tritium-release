#portions modified from https://personalpages.manchester.ac.uk/staff/paul.connolly/teaching/practicals/gaussian_plume_modelling.html


from shiny import App, ui, render, reactive
import numpy as np

# EPA ISC3 rural coefficients for sigma_y
ISC3_Y_COEFFICIENTS = {
    "A": (24.1670, 2.5334),
    "B": (18.3330, 1.8096),
    "C": (12.5000, 1.0857),
    "D": (8.3330,  0.72382),
    "E": (6.2500,  0.54287),
    "F": (4.1667,  0.36191),
}

# Tuples are (upper downwind distance in km, coefficient a, coefficient b)
ISC3_Z_COEFFICIENTS = {
    "A": [
        (0.10, 122.800, 0.94470),
        (0.15, 158.080, 1.05420),
        (0.20, 170.220, 1.09320),
        (0.25, 179.520, 1.12620),
        (0.30, 217.410, 1.26440),
        (0.40, 258.890, 1.40940),
        (0.50, 346.750, 1.72830),
        (3.11, 453.850, 2.11660),
        (np.inf, 5000.0, 0.0),  # sigma_z = 5000 m beyond 3.11 km
    ],
    "B": [
        (0.20, 90.673, 0.93198),
        (0.40, 98.483, 0.98332),
        (np.inf, 109.300, 1.09710),
    ],
    "C": [
        (np.inf, 61.141, 0.91465),
    ],
    "D": [
        (0.30, 34.459, 0.86974),
        (1.00, 32.093, 0.81066),
        (3.00, 32.093, 0.64403),
        (10.00, 33.504, 0.60486),
        (30.00, 36.650, 0.56589),
        (np.inf, 44.053, 0.51179),
    ],
    "E": [
        (0.10, 24.260, 0.83660),
        (0.30, 23.331, 0.81956),
        (1.00, 21.628, 0.75660),
        (2.00, 21.628, 0.63077),
        (4.00, 22.534, 0.57154),
        (10.00, 24.703, 0.50527),
        (20.00, 26.970, 0.46713),
        (40.00, 35.420, 0.37615),
        (np.inf, 47.618, 0.29592),
    ],
    "F": [
        (0.20, 15.209, 0.81558),
        (0.70, 14.457, 0.78407),
        (1.00, 13.953, 0.68465),
        (2.00, 13.953, 0.63227),
        (3.00, 14.823, 0.54503),
        (7.00, 16.187, 0.46490),
        (15.00, 17.836, 0.41507),
        (30.00, 22.651, 0.32681),
        (60.00, 27.074, 0.27436),
        (np.inf, 34.219, 0.21716),
    ],
}

def sigmas(x, stability):
    """
    Calculate EPA ISC3 rural dispersion coefficients.

    Parameters
    ----------
    x : float or array
        Downwind distance in meters.
    stability : str or int
        Pasquill-Gifford class A-F, or 1-6 where 1=A and 6=F.

    Returns
    -------
    sigma_y, sigma_z : float or arrays
        Dispersion coefficients in meters.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Downwind distance x must be greater than zero.")

    # Allow either A-F or 1-6
    stability = str(stability).strip().upper()
    stability = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "5": "E",
        "6": "F",
    }.get(stability, stability)

    if stability not in ISC3_Y_COEFFICIENTS:
        raise ValueError("Stability must be A-F or 1-6.")

    x_km = x / 1000.0 #convert downwind distance to km

    # Horizontal dispersion coefficient
    c, d = ISC3_Y_COEFFICIENTS[stability]

    theta = 0.017453293 * (c - d * np.log(x_km))

    sigma_y = (
        465.11628
        * x_km
        * np.tan(theta)
    )

    # Vertical dispersion coefficient
    sigma_z = np.empty_like(x_km)
    lower_bound = 0.0

    for upper_bound, a, b in ISC3_Z_COEFFICIENTS[stability]: #check
        mask = (
            (x_km > lower_bound)
            & (x_km <= upper_bound)
        )

        sigma_z[mask] = a * x_km[mask] ** b
        lower_bound = upper_bound

    # ISC3 caps sigma_z at 5000 m for classes A-C
    if stability in ("A", "B", "C"):
        sigma_z = np.minimum(sigma_z, 5000.0)

    return sigma_y, sigma_z





def gaussian_centerline(Q, u, x, z, h, stability):
    """
    Calculate Gaussian plume concentration on the centerline (y = 0)
    using EPA ISC3 rural dispersion coefficients.

    Parameters
    ----------
    Q : float
        Release rate in g/s.
    u : float
        Wind speed in m/s.
    x : float or array
        Downwind distance in meters.
    z : float
        Height above ground in meters.
    h : float
        Stack height in meters.
    stability : str or int
        Stability class A-F or 1-6.

    Returns
    -------
    C : float or array
        Concentration in g/L.
    """
    if u <= 0:
        raise ValueError("Wind speed must be greater than zero.")

    sigma_y, sigma_z = sigmas(
        x,
        stability
    )

    vertical_term = (
        np.exp(-((z - h) ** 2) / (2 * sigma_z**2))
        + np.exp(-((z + h) ** 2)/ (2 * sigma_z**2))
    )

    # Initial result is g/m^3
    concentration_g_m3 = (
        Q
        / (2 * np.pi * u * sigma_y * sigma_z)
        * vertical_term
    )

    # Convert g/m^3 to g/L
    return concentration_g_m3 / 1000

bq_to_g = 1/(1/3.016049*6.0221408e23*1.78238e-9)

AIR_LIMITS_BQL = { #define constants for reference lines in the air plot
    "Background": 2.82e-5*bq_to_g,
    "MDC (short duration)": 6.4e-2*bq_to_g,
    "MDC (long duration)": 4.0e-3*bq_to_g,
    "Max air concentration limit": 3.7*bq_to_g,
}

RIVER_LIMITS_GL = {
    "US tritium drinking water limit": 740 * bq_to_g,
    "T detectability in water": 0.06 * bq_to_g,
    "T background in water": (0.3 + 1) / 2 * bq_to_g,
}

# ---------- Air plume constants ----------
STACK_HEIGHT_M = 100.0
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
HOURS_PER_YEAR = 365 * 24

X_METERS = np.linspace(0.002, 10, 10000) * 1000

# Baseline release: 1 g/year, converted to g/s
Q_G_PER_S_BASE = 1.0 / SECONDS_PER_YEAR

# gaseous releases
FACILITY_GAS_RELEASES_GHR = {
"MSR (142 g/yr)":                [0.00823517,          0.01618787,          0.03985639],
"Fusion Power Plant (2.44 g/yr)": [3.68005323e-05,      2.781821e-04,      4.54854579e-03],
"HTGR (1.92x10⁻³ g/yr)":                [3.63710773e-07,      2.186352e-07,      5.12700729e-07],
"FHR (250 g/yr)":               [1.6012e-04,          0.02856893251845625,        9.412189e-02],
}



# GAS_BOUNDS_RATIOS = {
#     "PWR (7.14x10⁻³ g/yr)": [0.0, 8.151530e-07, 9.935326073021934e-06], #fix
#     "BWR (5.74×10⁻³ g/yr)": [1.320815383234566e-08, 6.552538e-07, 4.61766088728487e-06],
#     "HWR (0.512 g/yr)": [1.79926363e-05, 5.849676e-05, 1.39482924e-04],  # CANDU
# }


facilities = {
    "PWR": {
        "Q_g_year": 7.140741e-03,
        "color": "#e377c2",
    },
    "BWR": {
        "Q_g_year": 5.740023e-03,
        "color": "#ff7f0e",
    },
    "HWR": {
        "Q_g_year": 0.5124316399835502,
        "color": "#2ca02c",
    },
    "MSR": {
        "Q_g_year": 141.80575035400338,
        "color": "#17becf",
    },
    "Fusion Power Plant": {
        "Q_g_year": 2.4368751615272175,
        "color": "#9ecae1",
    },
    "HTGR": {
        "Q_g_year": 1.915245e-03,
        "color": "#9467bd",
    },
    "FHR": {
        "Q_g_year": 250.26384886167673,
        "color": "#bcbd22",
    },
}

def format_release_rate(value):
    """Formats release rates for display"""
    
    if value != 0 and abs(value) < 0.01:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exponent)

        return f"{mantissa:.3g}e{exponent:+03d}"

    return f"{value:.3g}"

# Use annual releases as multipliers for the 1 g/year baseline.
AIR_MAIN_FACTORS = {
    f"{name} ({format_release_rate(data['Q_g_year'])} g/yr)": data["Q_g_year"]
    for name, data in facilities.items()
}

AIR_FACILITY_LIST = list(AIR_MAIN_FACTORS.keys())

# Curve colors
FACILITY_COLORS = {
    "Fusion Power Plant (2.44 g/yr)":             "#9ecae1",
    "PWR (7.14x10⁻³ g/yr)":                            "#e377c2",
    "PWR (7.12×10⁻² g/yr)":                        "#e377c2",
    "BWR (2.10×10⁻³ g/yr)":                            "#ff7f0e",
    "BWR (5.74×10⁻³ g/yr)":                        "#ff7f0e",
    "HWR (0.512 g/yr)":                            "#2ca02c",
    "HWR (0.477 g/yr)":                                          "#2ca02c",
    "MSR (142 g/yr)":                            "#17becf",
    "HTGR (1.92x10⁻³ g/yr)":                      "#9467bd",
    "HTGR (5.84x10⁻² g/yr)":                           "#9467bd",
    "FHR (250 g/yr)":                            "#bcbd22",
}

# Match air curve colors to the labels generated above.
FACILITY_COLORS.update({
    f"{name} ({format_release_rate(data['Q_g_year'])} g/yr)": data["color"]
    for name, data in facilities.items()
})

# Reference-line colors (air)
COLOR_BG  = "#d62728"
COLOR_MDC = "#9467bd"

#ignore upper and lower bounds
LIQUID_RELEASES_BQPS = {
    "PWR (7.12×10⁻² g/yr)":  [0.0, 8.034277e+05, 0],
    "BWR (2.10×10⁻³ g/yr)":    [0.0, 2.373300e+04, 0],
    "HWR (0.477 g/yr)": [0, 5.387697e+06, 0],
    "HTGR (5.84x10⁻² g/yr)": [0, 6.591900e+05, 0],
    "MSR (142 g/yr)": [
        0,
        1.600295e+09,
        0
    ],

    "Fusion Power Plant (2.44 g/yr)": [
        0,
        2.750043e+07,
        0
    ],

    "FHR (250 g/yr)": [
        0,
        2.824258e+09,
        0
    ],
}


# River concentration from Bq/s:  C[Bq/L] = (Qi [Bq/s]) / (qr [m^3/s] * 1000 L/m^3)
def river_conc_bql(Qi_bqps, qr):
    return Qi_bqps / (qr * 1000.0) *bq_to_g

# ---------- Helpers ----------
def mph_to_mps(mph: float) -> float:
    return float(mph) * 0.44704 #convert miles per hour to meters per second

def stability_to_class(stab_choice: str) -> int: #get integers for stability class
    try:
        return int(stab_choice.split()[0])
    except Exception:
        return int(stab_choice)

def safe_for_log(y):
    y = np.array(y, dtype=float)
    y[y <= 0] = np.nan
    return y

def compute_base_profile(
    wind_mph: float,
    stab_class: int,
    stack_height_m: float = STACK_HEIGHT_M,
):
    """Return centerline (x_km, g/L) for a release of 1 g/year."""

    concentration_gl = gaussian_centerline(
        Q=Q_G_PER_S_BASE,
        u=mph_to_mps(wind_mph),
        x=X_METERS,
        z=1.0,
        h=stack_height_m,
        stability=stab_class,
    )

    return X_METERS / 1000.0, concentration_gl

# ---------- UI ----------
app_ui = ui.page_fluid(
    ui.h2("Tritium Concentration Explorer"),
    ui.navset_tab(
                # ----- Overview tab -----
        ui.nav_panel(
            "Overview",
            ui.h2("Overview"),
            ui.markdown(
                """
**Description**  

This web app visualizes resulting estimated tritium concentrations in the atmosphere and river for effluent releases from a 1 GWe nuclear power plant under standard operating conditions.  
It includes:  

- A **Gaussian plume model** for atmospheric transport of gaseous tritium releases.  
- A **river dilution model** for river transport liquid tritium releases.  

The goal is to contextualize tritium releases from future types of nuclear reactors, and provide a simple tool for visualizing resulting concentrations in any location.

---

**Key variables**  

*Atmosphere module*  

- **Facility type (air)** – select an annual gaseous tritium release rate for a specific type of nuclear facility.  
- **Wind speed (mph)** – used as the mean transport wind speed.  
- **Atmospheric stability class (A–F)** – controls plume spread and dilution.  
- **Stack height (m)** – emission height above ground.  
- **Distance (km)** – downwind centerline distance from the stack (0–10 km).  
- **Concentration (g/L)** – centerline concentration at 1 m above the ground (y=0, z=1). 

*River module*  

- **Facility type (river)** – select an annual liquid tritium release rate for a specific type of nuclear facility.  
- **River discharge qᵣ (m³/s)** – volumetric flow rate of the river.  
- **Concentration C (g/L)** – resulting concentration of the tritium in the river.  

---

**How to use this app**  

1. Use the **Atmosphere** tab to visualize how ground-level atmospheric concentration vs. distance changes with
   wind speed, stability class, stack height, and facility type.  
2. Use the **River** tab to explore how liquid releases dilute with increasing river discharge for different facility types.
3. If using your phone, select “Desktop site” in your browser settings to view the full version.
 


                """
            ),
        ),


        # ----- Atmosphere tab -----
        ui.nav_panel(
            "Atmosphere",
            ui.h2("Average Max Concentration vs. Distance for 1 GWe plant"),
            ui.markdown(
                "_For more information, please see "
                "https://hdl.handle.net/1721.1/159916_"
            ),
            ui.row(
                ui.column(
                    4,
                    ui.h3("Air: Gaussian plume"),
                    ui.input_numeric("wind_mph", "Average wind speed (mph)", value=7.582738531, min=0.001, step=0.1),
                    ui.input_select(
                        "stab", "Atmospheric stability (1–6 ≡ A–F):",
                        choices=[
                            "1 (A — very unstable)","2 (B — unstable)","3 (C — slightly unstable)",
                            "4 (D — neutral)","5 (E — slightly stable)","6 (F — stable)"
                        ],
                        selected="4 (D — neutral)",
                    ),

                    # Stack height is user-adjustable
                    ui.input_numeric("stack_h", "Stack height in m", value=100.0, min=0.0, step=1.0),



                    ui.input_checkbox_group(
                        "facilities", "Facilities to plot (air)", AIR_FACILITY_LIST, selected=AIR_FACILITY_LIST
                    ),

                    # Select the distance at which concentration will be reported
                    ui.input_numeric(
                        "query_distance_km",
                        "Show concentration at distance (km)",
                        value=1.0,
                        min=0.002,
                        max=10.0,
                        step=0.1,
                    ),

                    # Optional custom gas release
                    ui.input_checkbox("use_custom_release", "Add a custom gas release curve (g/hr)", value=False),
                    ui.panel_conditional(
                        "input.use_custom_release",
                        ui.input_numeric("custom_ghr", "Custom Q (g/hr)", value=0.01, min=0.0, step=0.001),
                        ui.input_text("custom_label", "Custom label", value="Custom (g/hr)")
                    ),

                    # ui.input_checkbox(
                    #     "show_bounds",
                    #     "Show air upper/lower bounds (MSR/FHR/HTGR/Fusion and PWR/BWR/HWR)",
                    #     value=False
                    # ),
                    ui.input_action_button("run", "Compute air"),
                    ui.hr(),
                    ui.markdown(
                        "**Notes**  \n"
                        "- Default **stack height 100 m** (user-adjustable).  \n"
                        "- Distance is along the **downwind centerline** (y=0).  \n"
                        "- **Wind direction is ignored** (fixed internally; not a user factor)."
                    ),
                    ui.output_text_verbatim("status", placeholder=True),
                    ui.output_text_verbatim("custom_debug", placeholder=True),

                    # Concentrations at the selected distance
                    ui.output_text_verbatim(
                        "concentration_at_distance",
                        placeholder=True,
                    ),



                    ui.output_text_verbatim("air_crossings", placeholder=True),
                ),
                ui.column(
                    8,
                    ui.output_plot("profile_plot", height="560px"),
                ),
            ),
        ),

        # ----- River tab -----
        ui.nav_panel(
            "River",
            ui.h2("River: Concentration vs. Discharge (Qi/qr) for a 1 GWe plant"),
            ui.row(
                ui.column(
                    4,
                    ui.input_numeric("qr_min", "Min discharge q_r (m³/s)", value=10.0, min=0.1, step=1.0),
                    ui.input_numeric("qr_max", "Max discharge q_r (m³/s)", value=1_000.0, min=1.0, step=10.0),
                    ui.input_checkbox_group(
                        "facilities_river",
                        "Facilities to include (river)",
                        choices=list(LIQUID_RELEASES_BQPS.keys()),
                        selected=list(LIQUID_RELEASES_BQPS.keys()),
                    ),
                    # ui.input_checkbox(
                    #     "show_river_bounds",
                    #     "Show river upper/lower bounds (PWR/BWR/HWR/HTGR)", value=False
                    # ),
                    ui.input_action_button("run_qr", "Compute river curves"),
                    ui.markdown("_C(q_r) = Qᵢ / (qᵣ·1000) in **g/L** — steady release, full mixing._"),
                    ui.output_text_verbatim("river_status", placeholder=True),

                    ui.output_text_verbatim("river_crossings", placeholder=True),
                    
                ),
                ui.column(
                    8,
                    ui.output_plot("river_qr_plot", height="520px"),
                ),
            ),
        ),
    ),
)


def first_crossing_distance_km(x_km, y, limit):
    """
    Returns the first downwind distance (km) where the curve crosses DOWN through `limit`.
    If it never exceeds the limit -> returns None.
    If it exceeds but never drops back below within range -> returns np.inf.
    Uses simple linear interpolation between grid points.
    """
    x_km = np.asarray(x_km, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(y)
    x_km = x_km[m]
    y = y[m]
    if x_km.size < 2:
        return None

    above = y >= limit
    if not np.any(above):
        return None  # never exceeds

    # Find first index where we go from above -> below
    idx = np.where(above[:-1] & (~above[1:]))[0]
    if idx.size == 0:
        return np.inf  # still above at 10 km

    i = int(idx[0])
    x0, x1 = x_km[i], x_km[i + 1]
    y0, y1 = y[i], y[i + 1]

    # Linear interpolation to y=limit
    if y1 == y0:
        return x0
    t = (limit - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)

def first_crossing_qr(qr_vals, conc_vals, limit):
    """
    Returns the first discharge q_r where the curve crosses DOWN through `limit`.
    If the curve never exceeds the limit, returns None.
    If it exceeds the limit everywhere in the plotted range, returns np.inf.
    Uses linear interpolation between neighboring points.
    """
    qr_vals = np.asarray(qr_vals, dtype=float)
    conc_vals = np.asarray(conc_vals, dtype=float)

    m = np.isfinite(conc_vals)
    qr_vals = qr_vals[m]
    conc_vals = conc_vals[m]

    if qr_vals.size < 2:
        return None

    above = conc_vals >= limit

    if not np.any(above):
        return None   # always below the threshold in plotted range

    idx = np.where(above[:-1] & (~above[1:]))[0]
    if idx.size == 0:
        return np.inf  # still above threshold at max plotted q_r

    i = int(idx[0])
    x0, x1 = qr_vals[i], qr_vals[i+1]
    y0, y1 = conc_vals[i], conc_vals[i+1]

    if y1 == y0:
        return x0

    t = (limit - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)



def first_up_crossing(x_km, concentration, limit):
    """
    Find distance where concentration rises
    from below the limit to at or above it.
    """

    # It is already above the limit at the first plotted distance.
    if concentration[0] >= limit:
        return -np.inf

    for i in range(len(concentration) - 1):
        if (
            concentration[i] < limit
            and concentration[i + 1] >= limit
        ):
            x0 = x_km[i]
            x1 = x_km[i + 1]
            c0 = concentration[i]
            c1 = concentration[i + 1]

            # Interpolate between the neighboring points.
            fraction = (limit - c0) / (c1 - c0)

            return x0 + fraction * (x1 - x0)

    # It never rises above the limit.
    return None



# server
def server(input, output, session):

    # air plume
    @reactive.Calc
    @reactive.event(input.run)
    def results_air():
        # gather wind/stack inputs
        wind_mph = float(input.wind_mph())
        stab = stability_to_class(input.stab())
        stack_h = float(input.stack_h())

        # Wind direction is not a factor; compute with fixed internal direction.
        x_km, base_gl = compute_base_profile(
            wind_mph, stab, stack_height_m=stack_h
        )

        selected = list(input.facilities()) or AIR_FACILITY_LIST
        curves = []
        for name in selected:
            factor = AIR_MAIN_FACTORS[name]  # annual release relative to 1 g/year
            y = base_gl * factor
            color = FACILITY_COLORS.get(name, "#7f7f7f")
            curves.append((name, color, x_km, y))

        # optional custom curve (extra; does not replace built-ins)
        if bool(input.use_custom_release()):
            q_ghr = float(input.custom_ghr())
            if q_ghr > 0:
                factor_custom = q_ghr * HOURS_PER_YEAR
                y_custom = base_gl * factor_custom
                label_custom = (input.custom_label() or "Custom (g/hr)").strip()
                color_custom = "#7f7f7f"  # neutral gray
                curves.append((label_custom, color_custom, x_km, y_custom))

        return curves
    
    @output
    @render.text
    @reactive.event(input.run)
    def air_crossings():
        wind_mph = float(input.wind_mph())
        stab = stability_to_class(input.stab())
        stack_h = float(input.stack_h())

        x_km, base_gl = compute_base_profile(wind_mph, stab, stack_height_m=stack_h)

        selected = list(input.facilities()) or AIR_FACILITY_LIST

        lines = []
        lines.append("Air threshold up-crossings and down-crossings:")
        lines.append(f"Wind = {wind_mph:.2f} mph | Stability = {stab} | Stack = {stack_h:.1f} m")
        lines.append("")

        for name in selected:
            factor = AIR_MAIN_FACTORS[name]
            y = base_gl * factor

            lines.append(name)
            for label, limit in AIR_LIMITS_BQL.items():
                up = first_up_crossing(x_km, y, limit)
                d = first_crossing_distance_km(x_km, y, limit)

                lines.append(f"  {label}:")

                if up is None:
                    lines.append("    Up-crossing: never rises above within plotted range")
                elif np.isneginf(up):
                    lines.append(f"    Up-crossing: already above at {x_km[0]:g} km")
                else:
                    lines.append(f"    Up-crossing: rises above at ~{up:.3f} km")

                if d is None:
                    lines.append("    Down-crossing: never exceeds within plotted range")
                elif d == np.inf:
                    lines.append("    Down-crossing: still above at 10 km")
                else:
                    lines.append(f"    Down-crossing: drops below at ~{d:.3f} km")

            lines.append("")

        return "\n".join(lines)

    @output
    @render.text
    def status():
        return (
            "Using EPA ISC3 rural Gaussian centerline "
            "(y=0, z=1 m). Click Compute air."
        )
    @output
    @render.text
    def custom_debug():
        if not bool(input.use_custom_release()):
            return "Custom curve not enabled."

        q_ghr = float(input.custom_ghr())
        if q_ghr <= 0:
            return "Custom release must be > 0."

        wind_mph = float(input.wind_mph())
        stab = stability_to_class(input.stab())
        stack_h = float(input.stack_h())

        x_km, base_gl = compute_base_profile(wind_mph, stab, stack_height_m=stack_h)
        y_custom = base_gl * q_ghr * HOURS_PER_YEAR
        c_1km = np.interp(1.0, x_km, y_custom)
        # c_0km = np.interp(0.0, x_km, y_custom) 

        return (
            f"Custom release: {q_ghr:.6g} g/hr\n"
            f"Concentration at 1 km: {c_1km:.3e} g/L\n"
            # f"Concentration at 0 km: {c_0km:.3e} g/L"
        )

    @output
    @render.text
    def concentration_at_distance():
        distance_km = float(input.query_distance_km())

        min_distance = X_METERS[0] / 1000.0
        max_distance = X_METERS[-1] / 1000.0

        if not min_distance <= distance_km <= max_distance:
            return (
                f"Distance must be between "
                f"{min_distance:.3f} and {max_distance:.1f} km."
            )

        curves = results_air()

        if not curves:
            return "No air facilities selected."

        lines = [
            f"Concentration at {distance_km:g} km from the source:",
            "",
        ]

        for name, color, x_km, concentration_gl in curves:
            valid = np.isfinite(x_km) & np.isfinite(concentration_gl)

            if np.count_nonzero(valid) < 2:
                lines.append(f"{name}: unavailable")
                continue

            concentration = np.interp(
                distance_km,
                x_km[valid],
                concentration_gl[valid],
            )

            lines.append(
                f"{name}\n"
                f"  {concentration:.3e} g/L-air\n"
            )

        return "\n".join(lines)



    @output
    @render.plot
    def profile_plot():
        import matplotlib.pyplot as plt
        import numpy as np

        show_bounds = False

        curves = results_air()
        fig, ax = plt.subplots(figsize=(9.6, 6.0)) #fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.set_xticks(np.arange(0, 10.5, 1)) #ax.set_xticks(np.arange(0, 10.5, 0.5))

        # # Force a narrower plotting area (left, bottom, width, height)
        # ax.set_position([0.15, 0.25, 0.5, 0.65])


        # Exact frame and smooth log plotting
        ax.set_xmargin(0); ax.margins(x=0, y=None)
        ax.set_xlim(0, 10)
        ax.set_yscale("log"); ax.set_ylim(1e-6*bq_to_g, 1e2*bq_to_g)
        ax.set_xlabel("Distance from Source (km)", fontsize=12)
        ax.set_ylabel("Concentration (g/L-air)", fontsize=12)
        ax.set_title("Average Max Concentration vs. Distance", fontsize=18, pad=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # show_bounds = bool(input.show_bounds())
        any_plotted = False

        for name, color, x_km, y_mid in curves:
            m = np.isfinite(y_mid)
            if not np.any(m):
                continue
            xv, yv_mid = x_km[m], y_mid[m]


            # Main (middle) curve
            ax.plot(xv, yv_mid, color=color, linewidth=2.4, label=name)
            any_plotted = True

            # if not show_bounds:
            #     continue

            # Bounds from gas g/hr triplets (MSR/FHR/HTGR/Fusion)
            # if name in FACILITY_GAS_RELEASES_GHR:
            if show_bounds and name in FACILITY_GAS_RELEASES_GHR:
                lo, mid, hi = FACILITY_GAS_RELEASES_GHR[name]
                if mid > 0:
                    lo_scale = lo / mid
                    hi_scale = hi / mid
                    yv_lo = safe_for_log(yv_mid * lo_scale)
                    yv_hi = safe_for_log(yv_mid * hi_scale)
                    ax.plot(xv, yv_lo, color=color, linestyle="--", linewidth=1.8, alpha=0.6)
                    ax.plot(xv, yv_hi, color=color, linestyle="--", linewidth=1.8, alpha=0.6)

            # Bounds from gaseous-normalized ratios for PWR/BWR/HWR
            # elif name in GAS_BOUNDS_RATIOS:
            elif show_bounds and name in GAS_BOUNDS_RATIOS:
                lo_r, mid_r, hi_r = GAS_BOUNDS_RATIOS[name]
                if mid_r > 0:
                    lo_scale = lo_r / mid_r
                    hi_scale = hi_r / mid_r
                    yv_lo = safe_for_log(yv_mid * lo_scale)
                    yv_hi = safe_for_log(yv_mid * hi_scale)
                    ax.plot(xv, yv_lo, color=color, linestyle="--", linewidth=1.8, alpha=0.6)
                    ax.plot(xv, yv_hi, color=color, linestyle="--", linewidth=1.8, alpha=0.6)

        # Reference lines
        ax.axhline(AIR_LIMITS_BQL["Background"], color=COLOR_BG,  linestyle="-",  linewidth=2, label="Background Tritium")
        ax.axhline(AIR_LIMITS_BQL["MDC (short duration)"], color=COLOR_MDC, linestyle="--", linewidth=2, label="MDC (short duration)")
        ax.axhline(AIR_LIMITS_BQL["MDC (long duration)"],  color=COLOR_MDC, linestyle=":",  linewidth=2, label="MDC (long duration)")
        ax.axhline(AIR_LIMITS_BQL["Max air concentration limit"], color="black", linestyle="-.", linewidth=2, label="Max air concentration limit")

        if any_plotted:
            # ax.legend(
            #     fontsize=10, ncol=2, loc="upper right",
            #     frameon=True, facecolor="white", edgecolor="black", framealpha=1.0
            # )
            fig.subplots_adjust(bottom=0.25)  # make room below the plot

            ax.legend(
                fontsize=10,
                ncol=3,                        
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),  # centered below the axes
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=1.0,
            )

        else:
            ax.text(0.5, 0.5, "No finite data to plot.\nTry different inputs.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)

        # plt.tight_layout()
        return fig

    # ---- River: C vs q_r curves (Qi in Bq/s) ----
    @reactive.Calc
    @reactive.event(input.run_qr)
    def river_curves():
        import numpy as np

        # Read user inputs
        qr_min = float(input.qr_min())
        qr_max = float(input.qr_max())

        # Basic sanity checks so the log scale doesn't explode
        if qr_min <= 0:
            qr_min = 1e-3
        if qr_max <= qr_min:
            qr_max = qr_min * 10.0

        # Build a smooth log-spaced grid between qr_min and qr_max
        qr_vals = np.logspace(np.log10(qr_min), np.log10(qr_max), 400)

        names = list(input.facilities_river()) or list(LIQUID_RELEASES_BQPS.keys())
        curves = []
        for name in names:
            low, mid, high = LIQUID_RELEASES_BQPS[name]  # Bq/s
            Qi_mid = mid
            C_mid = river_conc_bql(Qi_mid, qr_vals)
            color = FACILITY_COLORS.get(name, "#7f7f7f")
            curves.append((name, color, qr_vals, C_mid, low, mid, high))

        return curves


    @output
    @render.text
    def river_status():
        return "River model: Click Compute river curves."
    
    @output
    @render.text
    def river_crossings():
        curves = river_curves()

        if not curves:
            return "No river facilities selected."

        lines = []
        lines.append("River discharge where each plotted curve crosses each threshold:")
        lines.append("")

        for name, color, qr_vals, C_mid, low, mid, high in curves:
            lines.append(name)

            for label, limit in RIVER_LIMITS_GL.items():
                q_cross = first_crossing_qr(qr_vals, C_mid, limit)

                if q_cross is None:
                    lines.append(f"  {label}: never exceeds within plotted range")
                elif q_cross == np.inf:
                    lines.append(f"  {label}: still above at max plotted discharge")
                else:
                    lines.append(f"  {label}: crosses at q_r ≈ {q_cross:.3f} m^3/s")

            lines.append("")

        return "\n".join(lines)

    @output
    @render.plot
    def river_qr_plot():
        import matplotlib.pyplot as plt
        import numpy as np

        curves = river_curves()

        # Read the current min/max to set axis limits
        qr_min = float(input.qr_min())
        qr_max = float(input.qr_max())
        if qr_min <= 0:
            qr_min = 1e-3
        if qr_max <= qr_min:
            qr_max = qr_min * 10.0

        fig, ax = plt.subplots(figsize=(7, 5.2))

        if not curves:
            ax.text(0.5, 0.5, "No facilities selected.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            return fig

        # show_river_bounds = bool(input.show_river_bounds())

        for name, color, qr_vals, C_mid, low, mid, high in curves:
            # Dotted lines for MSR, Fusion, FHR
            if ("MSR" in name) or ("Fusion" in name) or ("FHR" in name):
                linestyle = ":"
                linewidth = 2.2
            else:
                linestyle = "-"
                linewidth = 2.4

            ax.plot(qr_vals, C_mid, color=color, linestyle=linestyle, linewidth=linewidth, label=name)


        ax.set_xmargin(0); ax.margins(x=0, y=None)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(qr_min, qr_max)
        ax.set_xlabel("River discharge q_r (m³/s)")
        ax.set_ylabel("Concentration C = Qᵢ/(qᵣ·1000) (g/L-H₂O)")
        ax.set_title("River Concentration vs. Discharge (Qi/qr)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # US drinking water limit
        ax.axhline(
            740*bq_to_g, color="black", linestyle="-.", linewidth=2,
            label="US tritium drinking water limit"
        )

        # T detectability in water
        ax.axhline(
            0.06*bq_to_g, color=COLOR_MDC, linestyle="--", linewidth=2,
            label="T detectability in water"
        )

        # T background in water
        ax.axhline(
            # 0.41*bq_to_g, color=COLOR_BG, linestyle="-", linewidth=2, # new one from https://www.sciencedirect.com/science/article/abs/pii/S0265931X16301576?via=ihub
            (0.3+1)/2*bq_to_g, color=COLOR_BG, linestyle="-", linewidth=2,
            label="T background in water"
            )
        

        fig.subplots_adjust(bottom=0.28)  # small extra space for footnote

        ax.legend(
            fontsize=10,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),   # slightly lower to avoid overlapping note
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0
        )

        ax.text(
            0.5, -0.19,
            "(MSR, fusion, FHR: estimated liquid release in case that all tritium is released as liquid effluents)",
            transform=ax.transAxes,
            ha="center",
            fontsize=10
        )

        plt.tight_layout()
        return fig


app = App(app_ui, server)
