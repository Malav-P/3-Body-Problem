from scipy.integrate import solve_ivp
import scipy.constants as sp
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = sp.value('Newtonian constant of gravitation')   # Gravitational Constant
m1 = 6e7                                            # mass of particle 1
m2 = 6e7                                            # mass of particle 2
m3 = 6e7                                            # mass of particle 3


#NOTE: if masses are kept the same, the resulting animation will have a centroid that does not move with time,
# this makes sense since the COM = geometric centroid if the masses are the same. COM does not move during in the
# absence of external forces.

M = m1 + m2 + m3                                    # total mass of system


y0 = [3, 1, 0, -2, 1, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # vector of 9 initial positions and 9 initial momenta
                                                               # z-components must be zero for in-plane solutions

xcm = (1 / M) * (m1 * y0[0] + m2 * y0[3] + m3 * y0[6])    # COM positions
ycm = (1 / M) * (m1 * y0[1] + m2 * y0[4] + m3 * y0[7])
zcm = (1 / M) * (m1 * y0[2] + m2 * y0[5] + m3 * y0[8])


def h(r1, r2, r3, p1, p2, p3):                            # Hamiltonian of the system
    -(G * m1 * m2) / la.norm(r1 - r2) - (G * m2 * m3) / la.norm(r2 - r3) - (G * m1 * m3) / la.norm(r1 - r3) + (
            la.norm(p1) ** 2) / (2 * m1) \
    + (la.norm(p2) ** 2) / (2 * m2) + (la.norm(p3) ** 2) / (2 * m3)


# system(t,y) is a function defining the first order system that will be numerically integrated. In other words,
# dy/dt = system(t,y) where y is a state vector of variables
def system(t, y):                     # defining the system for numerical integrator
    r1 = [None] * 3                   # position and momenta vectors for each particle
    r2 = [None] * 3
    r3 = [None] * 3
    p1 = [None] * 3
    p2 = [None] * 3
    p3 = [None] * 3

    for i in range(3):                # placing each position and momentum vector into one large state vector y
        r1[i] = y[i]
        r2[i] = y[3 + i]
        r3[i] = y[6 + i]
        p1[i] = y[9 + i]
        p2[i] = y[12 + i]
        p3[i] = y[15 + i]

    f = [None] * 18                   # initializing return variable of system.
    for i in range(18):               # this loop fills f with hamilton's equations of motion for the 3-body problem
        if i < 9:
            if i < 3:
                f[i] = y[i + 9] / m1
            elif i < 6:
                f[i] = y[i + 9] / m2
            else:
                f[i] = y[i + 9] / m3
        elif i < 12:
            f[i] = (-G * m1 * m2) * (y[i - 9] - y[i - 6]) * (la.norm(np.subtract(r1, r2)) ** -1.5) + (-G * m1 * m3) * (
                    y[i - 9] - y[i - 3]) * (la.norm(np.subtract(r1, r3)) ** -1.5)
        elif i < 15:
            f[i] = (G * m1 * m2) * (y[i - 12] - y[i - 9]) * (la.norm(np.subtract(r1, r2)) ** -1.5) + (-G * m2 * m3) * (
                    y[i - 9] - y[i - 6]) * (la.norm(np.subtract(r2, r3)) ** -1.5)
        else:
            f[i] = (G * m2 * m3) * (y[i - 12] - y[i - 9]) * (la.norm(np.subtract(r2, r3)) ** -1.5) + (G * m1 * m3) * (
                    y[i - 15] - y[i - 9]) * (la.norm(np.subtract(r1, r3)) ** -1.5)

    return f  # the return variable, dy/dt


T = 200  #  numerical integration length

sol = solve_ivp(system, [0, T], y0, dense_output=True)  # solve the initial value problem

t = np.linspace(0, T, T)               # from here on is plotting stuff
r1x = sol.sol(t)[0]
r1y = sol.sol(t)[1]
r1z = sol.sol(t)[2]
r2x = sol.sol(t)[3]
r2y = sol.sol(t)[4]
r2z = sol.sol(t)[5]
r3x = sol.sol(t)[6]
r3y = sol.sol(t)[7]
r3z = sol.sol(t)[8]

fig = plt.figure(figsize=(6, 6))  # Begin matplotlib stuff
ax = plt.subplot(111)

l1, = ax.plot([], [], color='red')
l2, = ax.plot([], [], color='green')
l3, = ax.plot([], [], color='blue')

dot1, = ax.plot([], [], 'o', color='black')
dot2, = ax.plot([], [], 'o', color='black')
dot3, = ax.plot([], [], 'o', color='black')

cm, = ax.plot([], [], 'x', color='black', markersize=3)

ls1, = plt.plot([], [], color='black', linewidth=0.7)
ls2, = plt.plot([], [], color='black', linewidth=0.7)
ls3, = plt.plot([], [], color='black', linewidth=0.7)

bs1, = plt.plot([], [], color='black', linewidth=0.5)
bs2, = plt.plot([], [], color='black', linewidth=0.5)
bs3, = plt.plot([], [], color='black', linewidth=0.5)

var = [True] * 3
for k in range(3):
    for i in range(6):
        if not y0[3 * i + k] == 0:
            var[k] = False

fp = r"D:\Users\malav\PycharmProjects\pythonProject1\animation.gif"   # ANIMATION SAVE PATH; change as needed
writergif = animation.PillowWriter(fps=30)

if var[2]:

    ax.set_xlim([xcm - 3.5, xcm + 3.5])
    ax.set_ylim([ycm - 3.5, ycm + 3.5])
    plt.xlabel('x')
    plt.ylabel('y')


    def animate(i):
        k = 70
        if i < k:
            l1.set_data(r1x[:i], r1y[:i])
            l2.set_data(r2x[:i], r2y[:i])
            l3.set_data(r3x[:i], r3y[:i])
        else:
            l1.set_data(r1x[i - k:i], r1y[i - k:i])
            l2.set_data(r2x[i - k:i], r2y[i - k:i])
            l3.set_data(r3x[i - k:i], r3y[i - k:i])

        dot1.set_data(r1x[i], r1y[i])
        dot2.set_data(r2x[i], r2y[i])
        dot3.set_data(r3x[i], r3y[i])

        cm.set_data(m1 * r1x[i] / M + m2 * r2x[i] / M + m3 * r3x[i] / M,
                    m1 * r1y[i] / M + m2 * r2y[i] / M + m3 * r3y[i] / M)

        ls1.set_data([r1x[i], r2x[i]], [r1y[i], r2y[i]])
        ls2.set_data([r2x[i], r3x[i]], [r2y[i], r3y[i]])
        ls3.set_data([r1x[i], r3x[i]], [r1y[i], r3y[i]])

        bs1.set_data([r1x[i], r2x[i] / 2 + r3x[i] / 2], [r1y[i], r2y[i] / 2 + r3y[i] / 2])
        bs2.set_data([r2x[i], r3x[i] / 2 + r1x[i] / 2], [r2y[i], r3y[i] / 2 + r1y[i] / 2])
        bs3.set_data([r3x[i], r1x[i] / 2 + r2x[i] / 2], [r3y[i], r1y[i] / 2 + r2y[i] / 2])


    anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=5)
    anim.save(fp, writer=writergif)
    plt.show()

elif var[0]:

    ax.set_xlim([ycm - 3.5, ycm + 3.5])
    ax.set_ylim([zcm - 3.5, zcm + 3.5])
    plt.xlabel('y')
    plt.ylabel('z')


    def animate(i):
        k = 40
        if i < k:
            l1.set_data(r1y[:i], r1z[:i])
            l2.set_data(r2y[:i], r2z[:i])
            l3.set_data(r3y[:i], r3z[:i])
        else:
            l1.set_data(r1y[i - k:i], r1z[i - k:i])
            l2.set_data(r2y[i - k:i], r2z[i - k:i])
            l3.set_data(r3y[i - k:i], r3z[i - k:i])

        dot1.set_data(r1y[i], r1z[i])
        dot2.set_data(r2y[i], r2z[i])
        dot3.set_data(r3y[i], r3z[i])

        cm.set_data(m1 * r1y[i] / M + m2 * r2y[i] / M + m3 * r3y[i] / M,
                    m1 * r1z[i] / M + m2 * r2z[i] / M + m3 * r3z[i] / M)

        ls1.set_data([r1y[i], r2y[i]], [r1z[i], r2z[i]])
        ls2.set_data([r2y[i], r3y[i]], [r2z[i], r3z[i]])
        ls3.set_data([r1y[i], r3y[i]], [r1z[i], r3z[i]])

        bs1.set_data([r1y[i], r2y[i] / 2 + r3y[i] / 2], [r1z[i], r2z[i] / 2 + r3z[i] / 2])
        bs2.set_data([r2y[i], r3y[i] / 2 + r1y[i] / 2], [r2z[i], r3z[i] / 2 + r1z[i] / 2])
        bs3.set_data([r3y[i], r1y[i] / 2 + r2y[i] / 2], [r3z[i], r1z[i] / 2 + r2z[i] / 2])


    anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=5)
    anim.save(fp, writer=writergif)
    plt.show()

elif var[1]:

    ax.set_xlim([xcm - 3.5, xcm + 3.5])
    ax.set_ylim([zcm - 3.5, zcm + 3.5])
    plt.xlabel('x')
    plt.ylabel('z')


    def animate(i):
        k = 40
        if i < k:
            l1.set_data(r1x[:i], r1z[:i])
            l2.set_data(r2x[:i], r2z[:i])
            l3.set_data(r3x[:i], r3z[:i])
        else:
            l1.set_data(r1x[i - k:i], r1z[i - k:i])
            l2.set_data(r2x[i - k:i], r2z[i - k:i])
            l3.set_data(r3x[i - k:i], r3z[i - k:i])

        dot1.set_data(r1x[i], r1z[i])
        dot2.set_data(r2x[i], r2z[i])
        dot3.set_data(r3x[i], r3z[i])

        cm.set_data(m1 * r1x[i] / M + m2 * r2x[i] / M + m3 * r3x[i] / M,
                    m1 * r1z[i] / M + m2 * r2z[i] / M + m3 * r3z[i] / M)

        ls1.set_data([r1x[i], r2x[i]], [r1z[i], r2z[i]])
        ls2.set_data([r2x[i], r3x[i]], [r2z[i], r3z[i]])
        ls3.set_data([r1x[i], r3x[i]], [r1z[i], r3z[i]])

        bs1.set_data([r1x[i], r2x[i] / 2 + r3x[i] / 2], [r1z[i], r2z[i] / 2 + r3z[i] / 2])
        bs2.set_data([r2x[i], r3x[i] / 2 + r1x[i] / 2], [r2z[i], r3z[i] / 2 + r1z[i] / 2])
        bs3.set_data([r3x[i], r1x[i] / 2 + r2x[i] / 2], [r3z[i], r1z[i] / 2 + r2z[i] / 2])


    anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=5)
    anim.save(fp, writer=writergif)
    plt.show()
else:
    print('solution cannot be plotted in a plane')
