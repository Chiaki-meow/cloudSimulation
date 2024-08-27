import taichi as ti

lin_iters = 20

N = 256
dx = 1.0 / N
dt = 0.1
# dt = 60  # 1 minute
diff = 0.001
visc = 0.001
force = 1e5
source = 100.0

dvel = False

v = ti.Vector.field(2, float, shape=(N + 2, N + 2), offset=(-1, -1))
v_prev = ti.Vector.field(2, float, shape=(N + 2, N + 2), offset=(-1, -1))
dens = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
dens_prev = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))

w = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
w_abs = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))

div = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
p = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
pixels = ti.field(float, shape=(N, N))

theta = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
qv = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
qc = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
qr = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))

qvs = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))

d_prev = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))

B = ti.Vector.field(2, float, shape=(N + 2, N + 2), offset=(-1, -1))

eps = 0.1
Er = 1.6
Ac = 0.003
Kc = 0.003

current_altitude = 8000

L = 2.5e3  # latent heat of vaporization

p0 = 101.325  # pressure at sea level

# parameters for temperature
z1 = 8000  # altitude for temperature
T_sea = 288.15  # temperature at sea level
T0 = 273.15  # temperature 0 degree

lapse_rate = 0.0065  # temperature lapse rate
lapse_rate0 = lapse_rate  # temperature lapse rate
lapse_rate1 = -lapse_rate  # temperature lapse rate

g = 9.81 * 1e-3  # gravity
R = 8314  # gas constant


# 初始化字段
@ti.kernel
def initialize_fields():
    for i, j in qv:
        qv[i, j] = 0.5  # 假设初始水蒸气混合比
        qc[i, j] = 0.5  # 假设初始云水混合比
        qr[i, j] = 0.0  # 假设初始雨水混合比
        theta[i, j] = 273.15  # 假设初始温度


@ti.func
def cal_pressure(altitude):
    return p0 * (1 - lapse_rate * altitude / T0) ** (g / (R * lapse_rate))


@ti.func
def cal_air_temperature(altitude):
    tmp = 0.0
    if altitude >= 0:
        if altitude <= z1:
            tmp = T0 + lapse_rate0 * altitude
        else:
            tmp = T0 + lapse_rate0 * z1 + lapse_rate1 * (altitude - z1)
    return tmp


Mair = 28.97  # molar mass of air
Mw = 18.02  # molar mass of water
# Xv = 0.622  # ratio of molar mass of water vapor to dry air
Xv = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
Mth = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
gemma_th = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))
c_pth = ti.field(float, shape=(N + 2, N + 2), offset=(-1, -1))


@ti.func
def cal_mth(_Xv):
    return _Xv * Mw + (1 - _Xv) * Mair  # molar mass of moist air


@ti.func
def cal_gemma_th(_Yv):
    return _Yv * 1.33 + (1 - _Yv) * 1.4


@ti.func
def cal_temp_rising_thermal(altitude, _Yv):
    return T_sea * (cal_pressure(altitude) / p0) ** ((cal_gemma_th(_Yv) - 1) / cal_gemma_th(_Yv))


@ti.func
def cal_buoyant(altitude, _Xv):
    return [0, g * ((Mair / cal_mth(_Xv)) * (cal_temp_rising_thermal(altitude, _Xv) / cal_air_temperature(altitude)) - 1)]


@ti.kernel
def add_source(a: ti.template(), b: ti.template()):
    for i, j in a:
        a[i, j] += dt * b[i, j]


@ti.kernel
def swap(a: ti.template(), b: ti.template()):
    for i, j in a:
        a[i, j], b[i, j] = b[i, j], a[i, j]


@ti.func
def set_bnd_2d(x: ti.template()):
    for i in range(N):
        x[-1, i] = x[0, i]
        x[N, i] = x[N - 1, i]
        x[i, -1] = x[i, 0]
        x[i, N] = x[i, N - 1]
        x[-1, i][0] *= -1.0
        x[N, i][0] *= -1.0
        x[i, -1][1] *= -1.0
        x[i, N][1] *= -1.0
    x[-1, -1] = (x[0, -1] + x[-1, 0]) / 2.0
    x[-1, N] = (x[0, N] + x[-1, N - 1]) / 2.0
    x[N, -1] = (x[N - 1, -1] + x[N, 0]) / 2.0
    x[N, N] = (x[N - 1, N] + x[N, N - 1]) / 2.0


@ti.func
def set_bnd(x: ti.template()):
    for i in range(N):
        x[-1, i] = x[0, i]
        x[N, i] = x[N - 1, i]
        x[i, -1] = x[i, 0]
        x[i, N] = x[i, N - 1]
    x[-1, -1] = (x[0, -1] + x[-1, 0]) / 2.0
    x[-1, N] = (x[0, N] + x[-1, N - 1]) / 2.0
    x[N, -1] = (x[N - 1, -1] + x[N, 0]) / 2.0
    x[N, N] = (x[N - 1, N] + x[N, N - 1]) / 2.0


@ti.func
def lin_solve(x: ti.template(), x0: ti.template(), a: float, c: float):
    for k in range(lin_iters):
        for i, j in ti.ndrange(N, N):
            x[i, j] = (x0[i, j] + a * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])) / c
        set_bnd(x)


@ti.kernel
def diffuse(a: ti.template(), a_prev: ti.template(), diff: float):
    k = dt * diff * N * N
    lin_solve(a, a_prev, k, 1.0 + 4.0 * k)


@ti.kernel
def advect(d: ti.template(), d0: ti.template(), v: ti.template()):
    dt0 = dt * N
    for i, j in ti.ndrange(N, N):
        x, y = i - dt0 * v[i, j][0], j - dt0 * v[i, j][1]
        if (x < 0.5): x = 0.5
        if (x > N + 0.5): x = N + 0.5
        i0, i1 = int(x), int(x) + 1
        if (y < 0.5): y = 0.5
        if (y > N + 0.5): y = N + 0.5
        j0, j1 = int(y), int(y) + 1
        s1, s0, t1, t0 = x - i0, i1 - x, y - j0, j1 - y
        d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    set_bnd(d)


@ti.kernel
def advect_for_q(d: ti.template(), v: ti.template()):
    dt0 = dt * N
    for i, j in ti.ndrange(N, N):
        x, y = i - dt0 * v[i, j][0], j - dt0 * v[i, j][1]
        if x < 0.5: x = 0.5
        if x > N + 0.5: x = N + 0.5
        i0, i1 = int(x), int(x) + 1
        if y < 0.5: y = 0.5
        if y > N + 0.5: y = N + 0.5
        j0, j1 = int(y), int(y) + 1
        s1, s0, t1, t0 = x - i0, i1 - x, y - j0, j1 - y
        d[i, j] = s0 * (t0 * d_prev[i0, j0] + t1 * d_prev[i0, j1]) + s1 * (t0 * d_prev[i1, j0] + t1 * d_prev[i1, j1])
    set_bnd(d)


@ti.kernel
def compute_vorticity():
    for i, j in ti.ndrange(N, N):
        dw_dx = 0.5 * (v[i + 1, j] - v[i - 1, j])
        dw_dy = 0.5 * (v[i, j + 1] - v[i, j - 1])
        w[i, j] = (dw_dx[1] - dw_dy[0])
        w_abs[i, j] = abs(w[i, j])


@ti.func
def vorticity_vec(i, j):
    w_grad_vec = ti.Vector(
        [0.5 * (w_abs[i + 1, j] - w_abs[i - 1, j]), 0.5 * (w_abs[i + 1, j] - w_abs[i - 1, j])]
    )
    w_grad_vec = w_grad_vec / w_grad_vec.norm()
    w_vec = ti.Vector([w_grad_vec[1], -w_grad_vec[0]]) * w[i, j]
    return ti.max(ti.min(w_vec, 0.1), -0.1)
    # return w_vec


@ti.kernel
def vorticity_confinement(v: ti.template()):
    for i, j in ti.ndrange(N, N):
        v[i, j] += dt * eps * vorticity_vec(i, j)


@ti.kernel
def compute_buoyancy():
    for i, j in ti.ndrange(N, N):
        B[i, j] = cal_buoyant(current_altitude, Xv[i, j])


@ti.func
def compute_external_force():
    return ti.Vector([0.0, 0.0])


@ti.kernel
def apply_buoyancy_and_external_force():
    for i, j in ti.ndrange(N, N):
        v[i, j] += dt * (B[i, j] + compute_external_force())


@ti.kernel
def project(a: ti.template(), a_prev: ti.template()):
    for i, j in ti.ndrange(N, N):
        div[i, j] = -(a[i + 1, j][0] - a[i - 1, j][0] + a[i, j + 1][1] - a[i, j - 1][1]) / (2.0 * N)
        p[i, j] = 0.0
    set_bnd(div)

    lin_solve(p, div, 1.0, 4.0)

    for i, j in ti.ndrange(N, N):
        a[i, j][0] -= N * (p[i + 1, j] - p[i - 1, j]) / 2.0
        a[i, j][1] -= N * (p[i, j + 1] - p[i, j - 1]) / 2.0
    set_bnd_2d(a)


def advect_quantities():
    swap(theta, d_prev)
    advect_for_q(theta, v)
    swap(qv, d_prev)
    advect_for_q(qv, v)
    swap(qc, d_prev)
    advect_for_q(qc, v)
    swap(qr, d_prev)
    advect_for_q(qr, v)


@ti.func
def get_celsius_temperature(_theta):
    return _theta - 273.15


@ti.func
def get_pressure_infinity(altitude):
    return p0 * 1000 * (1 - lapse_rate * altitude / T_sea) ** 5.2561


@ti.func
def cal_qvs(temperature, pressure):
    return 380.16 / pressure * ti.exp(17.67 * temperature / (temperature + 243.5))


@ti.kernel
def compute_qvs(_theta: ti.template()):
    for i, j in ti.ndrange(N, N):
        qvs[i, j] = cal_qvs(get_celsius_temperature(_theta[i, j]),
                            get_pressure_infinity(current_altitude))


@ti.kernel
def compute_qv():
    for i, j in ti.ndrange(N, N):
        qv[i, j] = qv[i, j] + min(qc[i, j], qvs[i, j] - qv[i, j]) + Er


@ti.kernel
def compute_qc():
    for i, j in ti.ndrange(N, N):
        qc[i, j] = qc[i, j] - min(qc[i, j], qvs[i, j] - qv[i, j]) - Ac - Kc


@ti.kernel
def compute_qr():
    for i, j in ti.ndrange(N, N):
        qr[i, j] = qr[i, j] + Kc + Ac - Er


@ti.func
def cal_xv(qi):
    return qi / (qi + 1)


@ti.kernel
def compute_xv(xv: ti.template(), qi: ti.template()):
    for i, j in ti.ndrange(N, N):
        xv[i, j] = cal_xv(qi[i, j])


@ti.kernel
def compute_mth(xv: ti.template()):
    for i, j in ti.ndrange(N, N):
        Mth[i, j] = cal_mth(xv[i, j])


@ti.kernel
def compute_gemma_th(yv: ti.template()):
    for i, j in ti.ndrange(N, N):
        yv[i, j] = cal_gemma_th(Xv[i, j] * Mw / Mth[i, j])


@ti.kernel
def get_heat_capacity(g_th: ti.template(), mth: ti.template()):
    for i, j in ti.ndrange(N, N):
        c_pth[i, j] = g_th[i, j] * R / (mth[i, j] * (g_th[i, j] - 1))


@ti.kernel
def compute_theta():
    for i, j in ti.ndrange(N, N):
        theta[i, j] += L / c_pth[i, j] * cal_xv(-min(qvs[i, j], qc[i, j]))


def dens_step():
    add_source(dens, dens_prev)
    swap(dens, dens_prev)
    advect(dens, dens_prev, v)
    swap(dens, dens_prev)
    diffuse(dens, dens_prev, diff)


def vel_step():
    add_source(v, v_prev)
    swap(v, v_prev)
    advect(v, v_prev, v_prev)
    swap(v, v_prev)
    diffuse(v, v_prev, visc)
    # project(v, v_prev)
    compute_vorticity()
    vorticity_confinement(v)
    compute_buoyancy()
    apply_buoyancy_and_external_force()

    project(v, v_prev)

    advect_quantities()
    compute_qvs(theta)
    compute_qv()
    compute_qc()
    compute_qr()

    compute_xv(Xv, qv)
    compute_mth(Xv)
    compute_gemma_th(gemma_th)
    get_heat_capacity(gemma_th, Mth)
    compute_theta()


def step():
    vel_step()
    dens_step()
