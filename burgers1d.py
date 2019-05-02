#!/usr/bin/python

#################################################
# 1D Burgers equation solver
# Sk. Mashfiqur Rahman
# Oklahoma State University
# CWID: A20102717
#################################################

import numpy as np
import matplotlib.pyplot as plt

# domain
x0 = 0.
xL = 1.

nx = 200
dx = (xL - x0)/float(nx)

x = np.empty(nx+1)
u = np.empty(nx+1)

for i in range(nx+1):
    x[i] = x0 + float(i)*dx

Tmax = 0.3
nt = 2000
nf = 20
kf = nt/nf
dt = Tmax/float(nt)
t = 0.

for i in range(nx+1):
    u[i] = np.sin(2.*np.pi*x[i])
u[0] = 0.
u[nx] = 0.
plot = [u]


def w3(a, b, c):
    eps = 1.e-6
    q1 = -0.5*a + 1.5*b
    q2 = 0.5*b + 0.5*c

    s1 = (b-a)**2
    s2 = (c-b)**2

    a1 = (1./3.)/(eps+s1)**2
    a2 = (2./3.)/(eps+s2)**2
    f = (a1*q1 + a2*q2)/(a1 + a2)

    return f


def rhs_weno(nx,dx,_u):
    q = np.empty(nx+3)
    r = np.empty(nx-1)
    for i in range(0, nx+1):
        q[i+1] = _u[i]
    q[0] = 2.*q[1] - q[2]
    q[nx+2] = 2.*q[nx+1] - q[nx]

    for i in range(1, nx):
        if _u[i] >= 0.:
            v1 = (q[i] - q[i-1])/dx
            v2 = (q[i+1] - q[i])/dx
            v3 = (q[i+2] - q[i+1])/dx

            g = w3(v1, v2, v3)
            r[i-1] = -_u[i]*g

        else:
            v1 = (q[i+3] - q[i+2])/dx
            v2 = (q[i+2] - q[i+1])/dx
            v3 = (q[i+1] - q[i])/dx

            g = w3(v1, v2, v3)
            r[i-1] = -_u[i]*g

    return r


def weno(nx, dx, dt, _u):
    v = np.empty(nx+1)
    v[0] = _u[0]
    v[nx] = _u[nx]

    r = rhs_weno(nx, dx, _u)
    for i in range(1,nx):
        v[i] = _u[i] + dt*r[i-1]

    r = rhs_weno(nx, dx, v)
    for i in range(1,nx):
        v[i] = 0.75*_u[i] + 0.25*v[i] + 0.25*dt*r[i-1]

    r = rhs_weno(nx, dx, v)
    for i in range(1,nx):
        v[i] = 1./3.*_u[i] + 2./3.*v[i] + 2./3.*dt*r[i-1]

    return v


# main function
for k in range(1, nt+1):

    u = weno(nx, dx, dt, u)
    t = t + dt

    if (k % kf) == 0:
        plot.append(u)
        # plt.figure()
        # plt.plot(x, u, label='t=final time')
        # plt.show()

plt.figure()
for i in range(nf+1):
    plt.plot(x, plot[i], linewidth=1.5, label=r't = '+str(i))
plt.ylabel('u')
plt.xlabel('x')
plt.legend(fontsize=10)
plt.tick_params(axis='both', labelsize=10)
plt.savefig('burgers1d.png', dpi = 1000)
plt.show()
