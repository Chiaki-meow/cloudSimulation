import taichi as ti

lin_iters = 20

N = 128
source_size = 40

vics_ = 0.0
diff_ = 0.0

# velocity field
u = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
u0 = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
v = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
v0 = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
w = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
w0 = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

# density field
dens = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
dens0 = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

# pressure field
div = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
p = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

# vorticity field
vor = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
curlx = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
curly = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
curlz = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

# stormscapes
theta_ = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))  # 单位是开尔文
qv = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
qc = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
qr = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
qvs = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
d_prev = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

Mair = 28.97  # molar mass of air
Mw = 18.02  # molar mass of water
Xv = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
Mth = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
gemma_th = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))
c_pth = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

B = ti.field(float, shape=(N + 2, N + 2, N + 2), offset=(-1, -1, -1))

eps = 0.00001
Er = 1.6
Ac = 0.0
Kc = 0.0

current_altitude = 8000
L = 2.5
p0 = 101.325

# parameters for temperature
z1 = 8000  # altitude for temperature
T_sea = 288.15  # temperature at sea level
T0 = 273.15  # temperature 0 degree

lapse_rate = 0.0065  # temperature lapse rate
lapse_rate0 = lapse_rate  # temperature lapse rate
lapse_rate1 = -lapse_rate  # temperature lapse rate

g = 9.81  # gravity
R = 8314  # gas constant

stagger = ti.Vector([0.5, 0.5, 0.5])

dx = 1.0 / N
dt = 0.1


@ti.kernel
def initialize_fields():
    for i, j, k in qv:
        qv[i, j, k] = 0.5  # 假设初始水蒸气混合比
        qc[i, j, k] = 0.5  # 假设初始云水混合比
        qr[i, j, k] = 0.0  # 假设初始雨水混合比
        theta_[i, j, k] = 273.15  # 假设初始温度


# for volume rendering
@ti.func
def sample(d):
    d = clamp(d)
    p_grid = d * N - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)
    return I


@ti.func
def clamp(d):
    for r in ti.static(range(d.n)):
        d[r] = min(1 - 1e-4 - dx + stagger[r] * dx, max(d[r], stagger[r] * dx))
    return d


@ti.kernel
def dens_reset():
    for I in ti.grouped(u):
        dens[I] = 0.0


# stable fluid
@ti.kernel
def init_scene():
    for i, j, k in ti.ndrange(source_size, source_size, source_size):
        # dens[i, j+source_size, k] += 50.0 * dt
        dens[i + (N - source_size) // 2, j + source_size, k + (N - source_size) // 2] += 100.0 * dt
    for i, j, k in ti.ndrange(N, N, N):
        if ti.random() > 0.9:
            u[i, j, k] += 2.0 * ti.random() - 1.0
            v[i, j, k] += 2.0 * ti.random() - 0.5
            w[i, j, k] += 2.0 * ti.random() - 1.0


@ti.kernel
def add_source(d: ti.template(), d0: ti.template()):
    for i, j, k in d:
        d[i, j, k] += dt * d0[i, j, k]


@ti.kernel
def swap(a: ti.template(), b: ti.template()):
    for i, j, k in a:
        a[i, j, k], b[i, j, k] = b[i, j, k], a[i, j, k]


@ti.func
def set_bnd(x: ti.template(), b: int):
    for i in range(N):
        for j in range(N):
            if b == 3:
                x[i, j, 0] = -x[i, j, 1]
                x[i, j, N + 1] = -x[i, j, N]
            else:
                x[i, j, 0] = x[i, j, 1]
                x[i, j, N + 1] = x[i, j, N]
    for i in range(N):
        for k in range(N):
            if b == 2:
                x[i, 0, k] = -x[i, 1, k]
                x[i, N + 1, k] = -x[i, N, k]
            else:
                x[i, 0, k] = x[i, 1, k]
                x[i, N + 1, k] = x[i, N, k]
    for j in range(N):
        for k in range(N):
            if b == 1:
                x[0, j, k] = -x[1, j, k]
                x[N + 1, j, k] = -x[N, j, k]
            else:
                x[0, j, k] = x[1, j, k]
                x[N + 1, j, k] = x[N, j, k]

    for i in range(N):
        x[i, 0, 0] = 1.0 / 2.0 * (x[i, 1, 0] + x[i, 0, 1])
        x[i, N + 1, 0] = 1.0 / 2.0 * (x[i, N, 0] + x[i, N + 1, 1])
        x[i, 0, N + 1] = 1.0 / 2.0 * (x[i, 0, N] + x[i, 1, N + 1])
        x[i, N + 1, N + 1] = 1.0 / 2.0 * (x[i, N, N + 1] + x[i, N + 1, N])

    for j in range(N):
        x[0, j, 0] = 1.0 / 2.0 * (x[1, j, 0] + x[0, j, 1])
        x[N + 1, j, 0] = 1.0 / 2.0 * (x[N, j, 0] + x[N + 1, j, 1])
        x[0, j, N + 1] = 1.0 / 2.0 * (x[0, j, N] + x[1, j, N + 1])
        x[N + 1, j, N + 1] = 1.0 / 2.0 * (x[N, j, N + 1] + x[N + 1, j, N])

    for k in range(N):
        x[0, 0, k] = 1.0 / 2.0 * (x[1, 0, k] + x[0, 1, k])
        x[N + 1, 0, k] = 1.0 / 2.0 * (x[N, 0, k] + x[N + 1, 1, k])
        x[0, N + 1, k] = 1.0 / 2.0 * (x[0, N, k] + x[1, N + 1, k])
        x[N + 1, N + 1, k] = 1.0 / 2.0 * (x[N, N + 1, k] + x[N + 1, N, k])

    x[0, 0, 0] = 1.0 / 3.0 * (x[1, 0, 0] + x[0, 1, 0] + x[0, 0, 1])
    x[0, N + 1, 0] = 1.0 / 3.0 * (x[1, N + 1, 0] + x[0, N, 0] + x[0, N + 1, 1])
    x[N + 1, 0, 0] = 1.0 / 3.0 * (x[N, 0, 0] + x[N + 1, 1, 0] + x[N + 1, 0, 1])
    x[N + 1, N + 1, 0] = 1.0 / 3.0 * (x[N, N + 1, 0] + x[N + 1, N, 0] + x[N + 1, N + 1, 1])

    x[0, 0, N + 1] = 1.0 / 3.0 * (x[1, 0, N + 1] + x[0, 1, N + 1] + x[0, 0, N])
    x[0, N + 1, N + 1] = 1.0 / 3.0 * (x[1, N + 1, N + 1] + x[0, N, N + 1] + x[0, N + 1, N])
    x[N + 1, 0, N + 1] = 1.0 / 3.0 * (x[N, 0, N + 1] + x[N + 1, 1, N + 1] + x[N + 1, 0, N])
    x[N + 1, N + 1, N + 1] = 1.0 / 3.0 * (x[N, N + 1, N + 1] + x[N + 1, N, N + 1] + x[N + 1, N + 1, N])


@ti.kernel
def advect(d: ti.template(), d0: ti.template(), u_: ti.template(), v_: ti.template(), w_: ti.template(), b: int):
    for i, j, k in d0:
        x = i - dt * u_[i, j, k]
        y = j - dt * v_[i, j, k]
        z = k - dt * w_[i, j, k]
        if x < 0.5:
            x = 0.5
        if x > N + 0.5:
            x = N + 0.5
        i0 = int(x)
        i1 = i0 + 1
        if y < 0.5:
            y = 0.5
        if y > N + 0.5:
            y = N + 0.5
        j0 = int(y)
        j1 = j0 + 1
        if z < 0.5:
            z = 0.5
        if z > N + 0.5:
            z = N + 0.5
        k0 = int(z)
        k1 = k0 + 1
        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1
        u1 = z - k0
        u0 = 1 - u1
        d[i, j, k] = s0 * (t0 * (u0 * d0[i0, j0, k0] + u1 * d0[i0, j0, k1]) +
                           t1 * (u0 * d0[i0, j1, k0] + u1 * d0[i0, j1, k1])) + \
                     s1 * (t0 * (u0 * d0[i1, j0, k0] + u1 * d0[i1, j0, k1]) +
                           t1 * (u0 * d0[i1, j1, k0] + u1 * d0[i1, j1, k1]))
    set_bnd(d, b)


@ti.func
def lin_solve(d: ti.template(), d0: ti.template(), a: float, c: float, b: int):
    for t in ti.static(range(lin_iters)):
        for i, j, k in d:
            d[i, j, k] = (d0[i, j, k] + a * (
                    d[i - 1, j, k] + d[i + 1, j, k] + d[i, j - 1, k] +
                    d[i, j + 1, k] + d[i, j, k - 1] + d[i, j, k + 1])
                          ) / c
        set_bnd(d, b)


@ti.kernel
def diffuse(d: ti.template(), d0: ti.template(), diff: float, b: int):
    a = dt * diff * N * N * N
    c = 1 + 6 * a
    lin_solve(d, d0, a, c, b)


@ti.kernel
def project(u_: ti.template(), v_: ti.template(), w_: ti.template(), p_: ti.template(), div_: ti.template()):
    for i, j, k in div_:
        div_[i, j, k] = -dx * (
                u_[i + 1, j, k] - u_[i - 1, j, k] +
                v_[i, j + 1, k] - v_[i, j - 1, k] +
                w_[i, j, k + 1] - w_[i, j, k - 1]
        ) / 3.0
        p_[i, j, k] = 0
    set_bnd(div_, 0)
    set_bnd(p_, 0)

    lin_solve(p_, div_, 1, 6, 0)

    for i, j, k in p_:
        u_[i, j, k] -= 0.33 * N * (p_[i + 1, j, k] - p_[i - 1, j, k])
        v_[i, j, k] -= 0.33 * N * (p_[i, j + 1, k] - p_[i, j - 1, k])
        w_[i, j, k] -= 0.33 * N * (p_[i, j, k + 1] - p_[i, j, k - 1])
    set_bnd(u_, 1)
    set_bnd(v_, 2)
    set_bnd(w_, 3)


@ti.kernel
def vorticity_confinement():
    x = 0.0
    y = 0.0
    z = 0.0
    for i, j, k in curlx:
        x = (w[i, j + 1, k] - w[i, j - 1, k] - v[i, j, k + 1] + v[i, j, k - 1]) * 0.5
        curlx[i, j, k] = x
        y = (u[i, j, k + 1] - u[i, j, k - 1] - w[i + 1, j, k] + w[i - 1, j, k]) * 0.5
        curly[i, j, k] = y
        z = (v[i + 1, j, k] - v[i - 1, j, k] - u[i, j + 1, k] + u[i, j - 1, k]) * 0.5
        curlz[i, j, k] = z
        vor[i, j, k] = ti.sqrt(x * x + y * y + z * z)

    for i, j, k in u:
        Nx = 0.5 * (vor[i + 1, j, k] - vor[i - 1, j, k])
        Ny = 0.5 * (vor[i, j + 1, k] - vor[i, j - 1, k])
        Nz = 0.5 * (vor[i, j, k + 1] - vor[i, j, k - 1])
        len = 1 / (ti.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-20)
        Nx *= len
        Ny *= len
        Nz *= len
        u[i, j, k] += dt * (Ny * curlz[i, j, k] - Nz * curly[i, j, k]) * eps
        v[i, j, k] += dt * (Nz * curlx[i, j, k] - Nx * curlz[i, j, k]) * eps
        w[i, j, k] += dt * (Nx * curly[i, j, k] - Ny * curlx[i, j, k]) * eps


@ti.func
def cal_mth(_Xv):
    return _Xv * Mw + (1 - _Xv) * Mair  # molar mass of moist air


@ti.func
def cal_gemma_th(_Yv):
    return _Yv * 1.33 + (1 - _Yv) * 1.4


@ti.func
def cal_pressure(altitude):
    return p0 * (1 - lapse_rate * altitude / T0) ** (g / (R * lapse_rate))


@ti.func
def cal_temp_rising_thermal(altitude, _Yv):
    return T_sea * (cal_pressure(altitude) / p0) ** ((cal_gemma_th(_Yv) - 1) / cal_gemma_th(_Yv))


@ti.func
def cal_air_temperature(altitude):
    tmp = 0.0
    if altitude >= 0:
        if altitude <= z1:
            tmp = T0 + lapse_rate0 * altitude
        else:
            tmp = T0 + lapse_rate0 * z1 + lapse_rate1 * (altitude - z1)
    return tmp


@ti.func
def cal_buoyant(altitude, Xv_ijk):
    return g * ((Mair / cal_mth(Xv_ijk)) * (
            cal_temp_rising_thermal(altitude, Xv_ijk) / cal_air_temperature(altitude)) - 1)


@ti.kernel
def compute_buoyancy():
    for i, j, k in B:
        B[i, j, k] = cal_buoyant(current_altitude, Xv[i, j, k])


@ti.kernel
def apply_force():
    for i, j, k in w:
        v[i, j, k] += dt * B[i, j, k]


def advect_quantities():
    swap(theta_, d_prev)
    advect(theta_, d_prev, u, v, w, 0)
    swap(qv, d_prev)
    advect(qv, d_prev, u, v, w, 0)
    swap(qc, d_prev)
    advect(qc, d_prev, u, v, w, 0)
    swap(qr, d_prev)
    advect(qr, d_prev, u, v, w, 0)


@ti.func
def get_celsius_temperature(_theta_):
    return _theta_ - 273.15


@ti.func
def get_pressure_infinity(altitude):
    return p0 * 1000 * (1 - lapse_rate * altitude / T_sea) ** 5.2561


# 摄氏度+帕斯卡
@ti.func
def cal_qvs(temperature, pressure):
    return 380.16 / pressure * ti.exp(17.67 * temperature / (temperature + 243.5))


@ti.kernel
def compute_qvs():
    for i, j, k in qvs:
        qvs[i, j, k] = cal_qvs(get_celsius_temperature(theta_[i, j, k]),
                               get_pressure_infinity(current_altitude))


@ti.kernel
def compute_qv():
    for i, j, k in qv:
        qv[i, j, k] = qv[i, j, k] + min(qc[i, j, k], qvs[i, j, k] - qv[i, j, k]) + Er


@ti.kernel
def compute_qc():
    for i, j, k in qc:
        qc[i, j, k] = qc[i, j, k] - min(qc[i, j, k], qvs[i, j, k] - qv[i, j, k]) - Ac - Kc


@ti.kernel
def compute_qr():
    for i, j, k in qr:
        qr[i, j, k] = qr[i, j, k] + Kc + Ac - Er


@ti.func
def cal_xv(qi):
    return qi / (qi + 1)


@ti.kernel
def compute_xv():
    for i, j, k in Xv:
        Xv[i, j, k] = cal_xv(qv[i, j, k])


@ti.kernel
def compute_mth():
    for i, j, k in Mth:
        Mth[i, j, k] = cal_mth(Xv[i, j, k])


@ti.kernel
def compute_gemma_th():
    for i, j, k in gemma_th:
        gemma_th[i, j, k] = cal_gemma_th(Xv[i, j, k] * Mw / Mth[i, j, k])


@ti.kernel
def get_heat_capacity():
    for i, j, k in c_pth:
        c_pth[i, j, k] = gemma_th[i, j, k] * R / (Mth[i, j, k] * (gemma_th[i, j, k] - 1))


@ti.kernel
def compute_theta():
    for i, j, k in theta_:
        theta_[i, j, k] += L / c_pth[i, j, k] * cal_xv(-min(qvs[i, j, k], qc[i, j, k]))


def dens_step():
    init_scene()
    swap(dens, dens0)
    advect(dens, dens0, u, v, w, 0)
    swap(dens, dens0)
    diffuse(dens, dens0, diff_, 0)


def vel_step():
    swap(u, u0)
    swap(v, v0)
    swap(w, w0)

    add_source(u, u0)
    add_source(v, v0)
    add_source(w, w0)

    swap(u, u0)
    swap(v, v0)
    swap(w, w0)

    advect(u, u0, u, v, w, 1)
    advect(v, v0, u, v, w, 2)
    advect(w, w0, u, v, w, 3)

    swap(u, u0)
    diffuse(u, u0, vics_, 1)
    swap(v, v0)
    diffuse(v, v0, vics_, 2)
    swap(w, w0)
    diffuse(w, w0, vics_, 3)

    # vorticity_confinement()
    compute_buoyancy()
    apply_force()

    project(u, v, w, p, div)

    advect_quantities()
    compute_qvs()
    compute_qv()
    compute_qc()
    compute_qr()

    compute_xv()
    compute_mth()
    compute_gemma_th()

    get_heat_capacity()
    compute_theta()


def step():
    print(B[30, 30, 30])
    dens_step()
    vel_step()
