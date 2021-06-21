# stopy

---
# Description:
Ultimately, the library aims to implement basic stochastic algorithms and pseudo-random number generators derived from non-standard, but often occurring, distributions that are not found in basic libraries. The library is based on the numpy library
        
---
# File list:
1.   processes.py
2.   dif_equ.py
3.   montecarlo.py
4.   "to be done" generate.py

---

# Instalation:
Pycharm:

        pip install numpy
        python -m pip install -e git+https://github.com/MarcinBaranek/stopy.git#egg=stopy
        
Google colab:

        !pip install -e git+https://github.com/MarcinBaranek/stopy.git#egg=stopy
        import sys
        sys.path.append("/content/src/stopy")

---
# Abstract Files
## processes.py
Contain classes with methods generating process trajectories. It currently includes the Poison, Wiener and Ornstein-Uhlenbeck processes
## dif_equ.py
includes the ItoProcess class along with the fit_test method which checks how well the process describes the data. Contains the EulerScheme class used to generate the trajectories of a solution to a stochastic differential equation. TBD MilsteinSchema
## montecarlo.py
Includes a Monte Carlo simulation class to approximate integrals or averaging random algorithms. includes methods for calculating empirical variance and asymptotic confidence intervals. It uses multiprocessing to speed up calculations. TBD update
## Abstract generate.py
TBD classes generating pseudo-random numbers from unusual distributions, such as from a sphere, a ball and others that I did not find in standard libraries

---

# Usage examples
## processes.py
        #generate Wiener process to plot
        to_plot = Wiener.generate_array(np.zeros(shape=(3,)), x_axis=True)
        plt.plot(to_plot[0], to_plot[1])
        plt.show()

        # generate process on grid
        grid = np.array([0.0, 1.9, 10.0, 30.0, 30.1, 32])
        plt.plot(grid,
            Wiener.generate_on_grid(grid, start_point=np.zeros(shape=(4,))))
        plt.show()

        # do something until the process is in unit 5 dimensional ball
        trace = Wiener.generate(np.zeros(shape=(5,)))
        point = trace.__next__()
        do_something(point)
        while (point ** 2).sum() < 1:
            point = trace.__next__()
            do_something(point)

## dif_equ.py

## montecarlo.py

## Abstract generate.py
