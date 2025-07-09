import numpy as np
import scipy as sc
import qutip as q
import matplotlib.pyplot as plt
from numpy import pi
import sympy as sy 
from IPython.display import display, Math
from pylab import rcParams

import multiprocessing as mp
import concurrent.futures


import scipy.optimize as opt
from random import uniform
from numpy import pi
from functools import partial

import scipy.io as sio



def build_J(N_a):
    Jx = q.spin_Jx(N_a)
    Jy = q.spin_Jy(N_a)
    Jz = q.spin_Jz(N_a)
    
    Jxx = Jx * Jx
    Jyy = Jy * Jy
    Jzz = Jz * Jz
    
    Jxyyx = Jx * Jy + Jy * Jx
    Jxzzx = Jx * Jz + Jz * Jx
    Jyzzy = Jy * Jz + Jz * Jy
    
    Jzzz = Jzz * Jz
    
    
    J = {'x' : Jx, 'y' : Jy, 'z' : Jz, 'x^2' : Jxx, 'y^2' : Jyy, 'z^2' : Jzz, 'xy + yx' : Jxyyx, 'xz + zx' : Jxzzx, 'yz + zy' : Jyzzy, 'z^3' : Jzzz }
    return J


def build_expmJ(J):
    Jxpi2 = (1j * J['x'] * pi/2).expm()
    Jxpi4 = (1j * J['x'] * pi/4).expm()
    Jx_pi2 = (-1j * J['x'] * pi/2).expm()
    Jx_pi4 = (-1j * J['x'] * pi/4).expm()
    
    Jypi2 = (1j * J['y'] * pi/2).expm()
    Jypi4 = (1j * J['y'] * pi/4).expm()
    Jy_pi2 = (-1j * J['y'] * pi/2).expm()
    Jy_pi4 = (-1j * J['y'] * pi/4).expm()
    
    Jzpi2 = (1j * J['z'] * pi/2).expm()
    Jzpi4 = (1j * J['z'] * pi/4).expm()
    Jz_pi2 = (-1j * J['z'] * pi/2).expm()
    Jz_pi4 = (-1j * J['z'] * pi/4).expm()
    
    expmJ = {'xpi/2' : Jxpi2, 'xpi/4' : Jxpi4, '-xpi/2' : Jx_pi2, '-xpi/4' : Jx_pi4, 'ypi/2' : Jypi2, 'ypi/4' : Jypi4, '-ypi/2' : Jy_pi2, '-ypi/4' : Jy_pi4, 'zpi/2' : Jzpi2, 'zpi/4' : Jzpi4, '-zpi/2' : Jz_pi2, '-zpi/4' : Jz_pi4}
    return expmJ


def transform_operators(J,state):
    j_x = (state.dag() * J['x'] * state)[0][0][0].real
    j_y = (state.dag() * J['y'] * state)[0][0][0].real
    j_z = (state.dag() * J['z'] * state)[0][0][0].real
    
    # print('Mean values of cubic state')
    # print('< J_x > = ', (state.dag() * J['x'] * state)[0][0][0].real)
    # print('< J_y > = ', (state.dag() * J['y'] * state)[0][0][0].real)
    # print('< J_z > = ', (state.dag() * J['z'] * state)[0][0][0].real)
    # print('\n')
    
    if j_x < 0:
        state_r = (-1j * pi/2 * J['y']).expm() * state
        j_x = (state_r.dag() * J['x'] * state_r)[0][0][0].real
        j_y = (state_r.dag() * J['y'] * state_r)[0][0][0].real
        j_z = (state_r.dag() * J['z'] * state_r)[0][0][0].real

    elif j_y < 0:
        state_r = (-1j * pi/2 * J['z']).expm() * state
        j_x = (state_r.dag() * J['x'] * state_r)[0][0][0].real
        j_y = (state_r.dag() * J['y'] * state_r)[0][0][0].real
        j_z = (state_r.dag() * J['z'] * state_r)[0][0][0].real

    elif j_z < 0:
        state_r = (-1j * pi/2 * J['x']).expm() * state
        j_x = (state_r.dag() * J['x'] * state_r)[0][0][0].real
        j_y = (state_r.dag() * J['y'] * state_r)[0][0][0].real
        j_z = (state_r.dag() * J['z'] * state_r)[0][0][0].real
    else:
        state_r = state
        
    # print('Mean values of cubic state - first rotation')
    # print('< J_x > = ', (state.dag() * J['x'] * state)[0][0][0].real)
    # print('< J_y > = ', (state.dag() * J['y'] * state)[0][0][0].real)
    # print('< J_z > = ', (state.dag() * J['z'] * state)[0][0][0].real)
    # print('\n')  
        
        
    phi, theta = sy.symbols('phi theta')
    x, y, z = sy.symbols('x y z')

    Rx = sy.Matrix([[1, 0, 0], [0, sy.cos(phi), -sy.sin(phi)], [0, sy.sin(phi), sy.cos(phi)]])
    Ry = sy.Matrix([[sy.cos(theta), 0, sy.sin(theta)], [0, 1, 0], [-sy.sin(theta), 0, sy.cos(theta)]])
    Rz = sy.Matrix([[sy.cos(phi), sy.sin(phi), 0], [-sy.sin(phi), sy.cos(phi), 0 ], [0, 0, 1]])

    v1 = sy.Matrix([[x],[y],[z]])

    R = Ry * Rz

    vout = R * v1

    s = (x, y, z, phi, theta)
    g_func = sy.lambdify(s, vout, modules='numpy')
    
    
    theta = np.arccos(j_x/(np.sqrt(j_x**2 + j_y**2))) 
    psi = np.arcsin(j_z / (np.sqrt(j_x**2 + j_y**2 + j_z**2))) 
    
    J_X = q.Qobj(g_func(J['x'], J['y'], J['z'], theta, psi)[0][0])
    J_Y = q.Qobj(g_func(J['x'], J['y'], J['z'], theta, psi)[1][0])
    J_Z = q.Qobj(g_func(J['x'], J['y'], J['z'], theta, psi)[2][0])
    

    j__x = (state_r.dag() * J_X * state_r)[0][0][0].real
    j__y = (state_r.dag() * J_Y * state_r)[0][0][0].real
    j__z = (state_r.dag() * J_Z * state_r)[0][0][0].real
    
    print('Mean values of cubic state - first transformation')
    print('< J_x > = ', j__x)
    print('< J_y > = ', j__y)
    print('< J_z > = ', j__z)
    print('\n')  
    
    Cov_bef = q.covariance_matrix([J_Y,J_Z],state_r*state_r.dag())
    
    # print('Covariance matrix - before last step')
    # print(Cov_bef[0])
    # print(Cov_bef[1])
    # print('\n')
    
    a = Cov_bef[0][0]
    b = Cov_bef[0][1]
    c = Cov_bef[1][0]
    d = Cov_bef[1][1]
    
    
    if (b < 10**(-10)) and (b > -10**(-10)) :
        b = 0
        c = 0
    
    if b == 0: 
        
        print('\n')
        print('Without rotation')
        print('\n')
        
        return[J_X, J_Y, J_Z, state_r]
        
    else:
        
        [eigenvector, eigenstate] = q.Qobj(Cov_bef).eigenstates()
        
        v1 = eigenstate[0]
        
        v2 = eigenstate[1]
        
        v = np.array([[v1[0][0][0].real, v2[0][0][0].real ], [v1[1][0][0].real, v2[1][0][0].real ]])     
          
        
        J__Y = (v[0][0] * J_Y + v[0][1] * J_Z)
        J__Z = (v[1][0] * J_Y + v[1][1] * J_Z)
    
        j__x = (state_r.dag() * J_X * state_r)[0][0][0].real
        j__y = (state_r.dag() * J__Y * state_r)[0][0][0].real
        j__z = (state_r.dag() * J__Z * state_r)[0][0][0].real
        
        print('\n')
        print('With rotation')
        print('\n')
                
        print('Mean values of cubic state - last transformation')
        print('< J_x > = ', j__x)
        print('< J__y > = ', j__y)
        print('< J__z > = ', j__z)
        print('\n')
        
        cov_cov = q.covariance_matrix([J__Y,J__Z],state_r*state_r.dag())
        
        # print(cov_cov)
        
        return[J_X, J__Y, J__Z, state_r]


# def Calc_var(J, out, t_c):
    
#     [J_X,J_Y,J_Z, state] = transform_operators(J,out)
    
#     O_n = (-1j * J_Y**3 * t_c).expm() * (J_Z) * (1j * J_Y**3 * t_c).expm()
#     var_y_minus = (state.dag() * O_n**2 * state) - (state.dag() * O_n * state)**2
#     print('Nonlinear squeezing (y_minus)')
#     display(Math('\zeta = {}'.format(round(var_y_minus[0][0][0].real,4))))

#     O_n = (1j * J_Y**3 * t_c).expm() * (J_Z) * (-1j * J_Y**3 * t_c).expm()
#     var_y_plus = (state.dag() * O_n**2 * state) - (state.dag() * O_n * state)**2
#     print('Nonlinear squeezing (y_plus)')
#     display(Math('\zeta = {}'.format(round(var_y_plus[0][0][0].real,4))))


#     O_n = (-1j * J_Z**3 * t_c).expm() * (J_Y) * (1j * J_Z**3 * t_c).expm()
#     var_z_minus = (state.dag() * O_n**2 * state) - (state.dag() * O_n * state)**2
#     print('Nonlinear squeezing (z_minus)')
#     display(Math('\zeta = {}'.format(round(var_z_minus[0][0][0].real,4))))

#     O_n = (1j * J_Z**3 * t_c).expm() * (J_Y) * (-1j * J_Z**3 * t_c).expm()
#     var_z_plus = (state.dag() * O_n**2 * state) - (state.dag() * O_n * state)**2
#     print('Nonlinear squeezing (z_plus)')
#     display(Math('\zeta = {}'.format(round(var_z_plus[0][0][0].real,4))))

#     return[[J_X, J_Y, J_Z], state, [ var_y_minus, var_y_plus, var_z_minus, var_z_plus]]







def calc_y_deviation(x, y):    
    if x>0 and y>0:
        # print('pp')
        phi =  np.arctan2(y,x) 
        return phi
    if x>0 and y<0:
        # print('pm')
        phi = np.arctan2(y,x)
        return phi
    if x<0 and y<0:
        # print('mm')
        phi = - (pi - np.arctan2(y,x))
        return phi
    if x<0 and y>0:
        # print('mp')
        phi = - (pi -np.arctan2(y,x))
        return phi

def calc_z_deviation(z, x):    
    if z>0 and x>0:
        # print('pp')
        theta =  -(pi/2 - np.arctan2(x,z))
        return theta
    if z>0 and x<0:
        # print('pm')
        theta = -(pi/2 - np.arctan2(x,z));
        return theta
    if z<0 and x<0:
        # print('mm')
        theta = - (pi/2 - np.arctan2(x,z))
        return theta
    if z<0 and x>0:
        # print('mp')
        theta = - (pi/2 - np.arctan2(x,z))
        return theta

def rotation_calibration(Input_state, J):
        
    state_to_correction = Input_state
    jx = (state_to_correction.dag() * J['x'] * state_to_correction)[0][0][0].real
    jy = (state_to_correction.dag() * J['y'] * state_to_correction)[0][0][0].real
    jz = (state_to_correction.dag() * J['z'] * state_to_correction)[0][0][0].real
    # print(jx,jy,jz)
    # print('Input')
    # print('     <J_x> = ', jx)
    # print('     <J_y> = ', jy)
    # print('     <J_z> = ', jz)
    # print('\n')
    
    if jx > (-10**(-7)) and jx < (10**(-7)):
        jx = 0
    if jy > (-10**(-7)) and jy < (10**(-7)):
        jy = 0
    if jz > (-10**(-7)) and jz < (10**(-7)):
        jz = 0
    # print(jx,jy,jz)
    
    if jz == 0 and jy == 0 and jx!=0:

        to_the_zero_j_z = state_to_correction


    else:
        # print('tady?')
        # if jx > (-10**(-8)) and jx < (10**(-8)):
        #     jx = 0
        # if jy > (-10**(-8)) and jy < (10**(-8)):
        #     jy = 0

        if jx == 0:
            # print('tato cesta')
            if jz > 0: 
                state_to_correction = (-1j * J['y'] * pi/2).expm() * state_to_correction
            else:
                state_to_correction = (1j * J['y'] * pi/2).expm() * state_to_correction 

            jx = (state_to_correction.dag() * J['x'] * state_to_correction)[0][0][0].real
            jy = (state_to_correction.dag() * J['y'] * state_to_correction)[0][0][0].real
            jz = (state_to_correction.dag() * J['z'] * state_to_correction)[0][0][0].real
            
            if jx > (-10**(-7)) and jx < (10**(-7)):
                jx = 0
            if jy > (-10**(-7)) and jy < (10**(-7)):
                jy = 0
            if jz > (-10**(-7)) and jz < (10**(-7)):
                jz = 0
            # print(jx, jy, jz)
        
        if jz == 0 and jy == 0 and jx!=0:
                if jx > 0: 
                    to_the_zero_j_z = state_to_correction
                else:
                    to_the_zero_j_z = (1j * J['y'] * pi) * state_to_correction
            
        if jy != 0 and jx!=0:
#             to_the_zero_j_z = state_to_correction
        
            aha = calc_y_deviation(jx.real, jy.real)
            # print(aha)
            to_the_zero_j_y = (1j * aha * J['z']).expm() * state_to_correction

            jx = (to_the_zero_j_y.dag() * J['x'] * to_the_zero_j_y)[0][0][0].real
            jy = (to_the_zero_j_y.dag() * J['y'] * to_the_zero_j_y)[0][0][0].real
            jz = (to_the_zero_j_y.dag() * J['z'] * to_the_zero_j_y)[0][0][0].real

            # print('First correction')
            # print('     <J_x> = ', jx)
            # print('     <J_y> = ', jy)
            # print('     <J_z> = ', jz)
            # print('\n')

            if jz == 0 and jy == 0 and jx!=0:

                to_the_zero_j_z = to_the_zero_j_y

            else:

                ehe = calc_z_deviation(jz.real, jx.real)

                to_the_zero_j_z = (1j * ehe * J['y']).expm() * to_the_zero_j_y

                jx = to_the_zero_j_z.dag() * J['x'] * to_the_zero_j_z
                jy = to_the_zero_j_z.dag() * J['y'] * to_the_zero_j_z
                jz = to_the_zero_j_z.dag() * J['z'] * to_the_zero_j_z

                # print('Last correction')
                # print('     <J_x> = ', jx[0][0][0].real)
                # print('     <J_y> = ', jy[0][0][0].real)
                # print('     <J_z> = ', jz[0][0][0].real)
                # print('\n')
        if jx != 0 and jz !=0:
            
            aha = calc_z_deviation(jz, jx)
            to_the_zero_j_y = (1j * aha * J['y']).expm() * state_to_correction
            
            jx = (to_the_zero_j_y.dag() * J['x'] * to_the_zero_j_y)[0][0][0].real
            jy = (to_the_zero_j_y.dag() * J['y'] * to_the_zero_j_y)[0][0][0].real
            jz = (to_the_zero_j_y.dag() * J['z'] * to_the_zero_j_y)[0][0][0].real
            # print(jx,jy,jz)
            
            if jz == 0 and jy == 0 and jx!=0:

                to_the_zero_j_z = to_the_zero_j_y
                
            else:

                ehe = calc_y_deviation(jx.real, jy.real)
                to_the_zero_j_z = (1j * ehe * J['z']).expm() * to_the_zero_j_y

                jx = to_the_zero_j_z.dag() * J['x'] * to_the_zero_j_z
                jy = to_the_zero_j_z.dag() * J['y'] * to_the_zero_j_z
                jz = to_the_zero_j_z.dag() * J['z'] * to_the_zero_j_z
            
    # print(jx,jy,jz)
    Cov_m =  q.covariance_matrix([J['y'],J['z']],to_the_zero_j_z*to_the_zero_j_z.dag())

    if Cov_m[0][1] == 0 and Cov_m[1][0] == 0 :
        return to_the_zero_j_z
    else:
        [eigenvector, eigenstate] = q.Qobj(Cov_m).eigenstates()

        final_final = (1j * np.arcsin((eigenstate[0])[0][0])[0] * J['x']).expm() * to_the_zero_j_z

        Cov =  q.covariance_matrix([J['y'],J['z']],final_final*final_final.dag())

        # print(Cov)

        if Cov[0][1] > (- 10 * 10**(-5)) and Cov[0][1] < (10 * 10**(-5)):
            Cov[0][1] = 0

        if Cov[1][0] > (10 * -1**(-5)) and Cov[1][0] < (10 * 10**(-5)):
            Cov[1][0] = 0

        if Cov[0][1] == 0 and Cov[1][0] == 0:
            return final_final
            # print(final_final)
        else:
            print(Cov)
            print('You are fucked')

def Calc_variance_3(state, t_c, J):
    
    
    O13 = (1j * t_c * J['z']**3).expm() * J['y'] * (-1j * t_c * J['z']**3).expm()  
    var13 = ((state.dag() * O13**2 * state) - (state.dag() * O13 * state)**2)[0][0][0]
    
    O23 = (-1j * t_c * J['z']**3).expm() * (J['y']) * (1j * t_c * J['z']**3).expm()
    var23 = ((state.dag() * O23**2 * state) - (state.dag() * O23 * state)**2)[0][0][0]
    
    O33 = (1j * t_c * (J['y']**3)).expm() * (J['z']) * (-1j * t_c * (J['y']**3)).expm()
    var33 = ((state.dag() * O33**2 * state) - (state.dag() * O33 * state)**2)[0][0][0]
    
    O43 = (-1j * t_c * (J['y']**3)).expm() * (J['z']) * (1j * t_c * (J['y']**3)).expm()
    var43 = ((state.dag() * O43**2 * state) - (state.dag() * O43 * state)**2)[0][0][0]
    
    if var13 == var23:
        O3 = np.array([var33, var43])
        Os3 = ['J_y J_z -J_y', '-J_y J_z J_y']
        answer = np.where(min(O3) == O3)[0][0]
    elif var33 == var43:
        O3 = np.array([var13, var23])
        Os3 = ['J_z J_y -J_z','-J_z J_y J_z']
        answer = np.where(min(O3) == O3)[0][0]
    
    # print(O)
    
    # print(Os3[answer])
    
    return O3[answer]
    
def Calc_variance_2(state, t_c, J):
    
    
    O12 = (1j * t_c * J['z']**2).expm() * J['y'] * (-1j * t_c * J['z']**2).expm()  
    var12 = ((state.dag() * O12**2 * state) - (state.dag() * O12 * state)**2)[0][0][0]
    
    O22 = (-1j * t_c * J['z']**2).expm() * (J['y']) * (1j * t_c * J['z']**2).expm()
    var22 = ((state.dag() * O22**2 * state) - (state.dag() * O22 * state)**2)[0][0][0]
    
    O32 = (1j * t_c * (J['y']**2)).expm() * (J['z']) * (-1j * t_c * (J['y']**2)).expm()
    var32 = ((state.dag() * O32**2 * state) - (state.dag() * O32 * state)**2)[0][0][0]
    
    O42 = (-1j * t_c * (J['y']**2)).expm() * (J['z']) * (1j * t_c * (J['y']**2)).expm()
    var42 = ((state.dag() * O42**2 * state) - (state.dag() * O42 * state)**2)[0][0][0]
    
    O2 = np.array([var12, var22, var32, var42])
    Os2 = ['J_z J_y -J_z','J_z -J_y -J_z', 'J_y J_z -J_y', 'J_y -J_z -J_y']
    answer2 = np.where(min(O2) == O2)[0][0]
    
#     print(O)
    
#     print(Os[answer])
    
    return O2[answer2]
    