import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from numpy import asarray
from numpy import savetxt
import os
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
start = time.time() #Saves the begining time of the process

#==============================================================================
#ESTIMACIONES DE BETA y OTRAS VARIABLES DEL MODELO SIR
#==============================================================================
def EstimacionVariables(acumuladosReportados, POSITIVOSDEPTO, Poblacion, gamma):
    #INICIALIZACION DE VARIABLES 
    A = acumuladosReportados #TOTAL DE INDIVIDUOS QUE SE HAN INFECTADO.
    dA = POSITIVOSDEPTO #CASOS NUEVOS REPORTADOS CADA DIA POR CADA DEPARTAMENTO
    R = np.zeros((len(A),len(A[0])))
    I = A - R
    S = np.zeros((len(A),len(A[0])))
    Beta = np.zeros((len(A),len(A[0])))
    Rt = np.zeros((len(A),len(A[0])))
    R0 = np.zeros((len(A),len(A[0])))

    for i in range(len(A)):
        for j in range(len(A[0])):
            S[i][j] = Poblacion[i] - A[i][j] #ECUACION (4.37)
                
    for i in range(len(A)):
        N = Poblacion[i]
        for j in range(len(A[0])-1):
            if j>20: #porque empezamos desde j=0.
                R[i][j] = dA[i][j-21] + R[i][j-1] #ECUACION (4.38)
            else:
                R[i][j] = 0
            
            if A[i][j] - R[i][j] <= 0:
                I[i][j] = I[i][j-1]
            else:
                I[i][j] = A[i][j] - R[i][j] #ECUACION (4.39)
            S[i][j] = N - I[i][j] - R[i][j]
            Beta[i][j] = dA[i][j+1]/(S[i][j]*I[i][j])*N ##ECUACION (5.1)
            #mas adelante quitar el ultimo de beta porque es cero.
            Rt[i][j] = Beta[i][j]/gamma * S[i][j]/N
            R0[i][j] = Beta[i][j]/gamma
        
            if A[i][j] <= 0:
                A[i][j] = A[i][j-1]
    return A, dA, S, I,R, Beta, Rt, R0
    
    
#######################################################    

def DifFinitas2(I0,S0,R0,BetaCoef,n,Poblacion,dA0,A0, gamma, t, dt):
    Infectados = [I0]
    Sucept = [S0] ##########OJO EL PRIMER RESULTADO DE LA SOLUCION ES IGUAL AL VALOR INICIAL, LO MODIFICAREMOS
    Removidos = [R0]
    RepDiario = [dA0]
    ACUMULADO = [A0]
    gamma_inv = int(1/gamma)
    TamanioPrueba = t #el tamanio final es TamanioPrueba + 1
    for j in range(TamanioPrueba):
#        dA1 = BetaCoef[j][0]*I0*S0/Poblacion[n] - gamma*I0 #por persona
        #no tengo que restarle los recuperados a los acumulados, debo restarlo en los infectados actuales
    #    dA1 = BetaCoef[j][0]*I0*S0 - gamma*I0
        dA1 = BetaCoef[j]*I0*S0/Poblacion[n]
        S1 = S0 - BetaCoef[j]*I0*S0/Poblacion[n]
        A1 = A0 + dA1
        R1 = R0 + gamma*I0
        I1 = I0 + dA1  -  gamma*I0 
        #if j > (gamma_inv-1):
        #    R1 = R0 + RepDiario[j-gamma_inv]
        #    I1 = I0 + dA1  - RepDiario[j-gamma_inv]
        #else:
        #    R1 = R0
        #    I1 = I0 + dA1
        Infectados.append(I1)
        Sucept.append(S1)
        Removidos.append(R1)
        RepDiario.append(dA1)
        ACUMULADO.append(A1)
        I0, S0, R0, A0 = I1, S1, R1, A1
    return Infectados, Sucept, Removidos, ACUMULADO, RepDiario

    
    
    
    
    
    
    
    
