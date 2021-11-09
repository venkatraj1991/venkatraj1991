import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import openseespy.opensees as op
import openseespy.postprocessing.Get_Rendering as opsl
import math
import sys

def get_pyParam(pyDepth, gamma, phiDegree, b, pEleLength, puSwitch, kSwitch, gwtSwitch):
    # ----------------------------------------------------------
    #  define ultimate lateral resistance, pult
    # ----------------------------------------------------------

    # pult is defined per API recommendations (Reese and Van Impe, 2001 or API, 1987) for puSwitch = 1
    #  OR per the method of Brinch Hansen (1961) for puSwitch = 2

    pi = 3.14159265358979
    phi = phiDegree * (pi / 180)
    zbRatio = pyDepth / b

    # -------API recommended method-------

    if puSwitch == 1:

        # obtain loading-type coefficient A for given depth-to-diameter ratio zb
        #  ---> values are obtained from a figure and are therefore approximate
        zb = []
        dataNum = 41
        for i in range(dataNum):
            b1 = i * 0.125
            zb.append(b1)
        As = [2.8460, 2.7105, 2.6242, 2.5257, 2.4271, 2.3409, 2.2546, 2.1437, 2.0575, 1.9589, 1.8973, 1.8111, 1.7372,
              1.6632, 1.5893, 1.5277, 1.4415, 1.3799, 1.3368, 1.2690, 1.2074, 1.1581,
              1.1211, 1.0780, 1.0349, 1.0164, 0.9979, 0.9733, 0.9610, 0.9487, 0.9363, 0.9117, 0.8994, 0.8994, 0.8871,
              0.8871, 0.8809, 0.8809, 0.8809, 0.8809, 0.8809]

        # linear interpolation to define A for intermediate values of depth:diameter ratio
        for i in range(dataNum):
            if zbRatio >= 5.0:
                A = 0.88
            elif zb[i] <= zbRatio and zbRatio <= zb[i + 1]:
                A = (As[i + 1] - As[i]) / (zb[i + 1] - zb[i]) * (zbRatio - zb[i]) + As[i]

        # define common terms
        alpha = phi / 2
        beta = pi / 4 + phi / 2
        K0 = 0.4

        tan_1 = math.tan(pi / 4 - phi / 2)
        Ka = math.pow(tan_1, 2)

        # terms for Equation (3.44), Reese and Van Impe (2001)
        tan_2 = math.tan(phi)
        tan_3 = math.tan(beta - phi)
        sin_1 = math.sin(beta)
        cos_1 = math.cos(alpha)
        c1 = K0 * tan_2 * sin_1 / (tan_3 * cos_1)

        tan_4 = math.tan(beta)
        tan_5 = math.tan(alpha)
        c2 = (tan_4 / tan_3) * tan_4 * tan_5

        c3 = K0 * tan_4 * (tan_2 * sin_1 - tan_5)

        c4 = tan_4 / tan_3 - Ka

        # terms for Equation (3.45), Reese and Van Impe (2001)
        pow_1 = math.pow(tan_4, 8)
        pow_2 = math.pow(tan_4, 4)
        c5 = Ka * (pow_1 - 1)
        c6 = K0 * tan_2 * pow_2

        # Equation (3.44), Reese and Van Impe (2001)
        pst = gamma * pyDepth * (pyDepth * (c1 + c2 + c3) + b * c4)

        # Equation (3.45), Reese and Van Impe (2001)
        psd = b * gamma * pyDepth * (c5 + c6)

        # pult is the lesser of pst and psd. At surface, an arbitrary value is defined
        if pst <= psd:
            if pyDepth == 0:
                pu = 0.01

            else:
                pu = A * pst

        else:
            pu = A * psd

        # PySimple1 material formulated with pult as a force, not force/length, multiply by trib. length
        pult = pu * pEleLength

    # -------Brinch Hansen method-------
    elif puSwitch == 2:
        # pressure at ground surface
        cos_2 = math.cos(phi)

        tan_6 = math.tan(pi / 4 + phi / 2)

        sin_2 = math.sin(phi)
        sin_3 = math.sin(pi / 4 + phi / 2)

        exp_1 = math.exp((pi / 2 + phi) * tan_2)
        exp_2 = math.exp(-(pi / 2 - phi) * tan_2)

        Kqo = exp_1 * cos_2 * tan_6 - exp_2 * cos_2 * tan_1
        Kco = (1 / tan_2) * (exp_1 * cos_2 * tan_6 - 1)

        # pressure at great depth
        exp_3 = math.exp(pi * tan_2)
        pow_3 = math.pow(tan_2, 4)
        pow_4 = math.pow(tan_6, 2)
        dcinf = 1.58 + 4.09 * (pow_3)
        Nc = (1 / tan_2) * (exp_3) * (pow_4 - 1)
        Ko = 1 - sin_2
        Kcinf = Nc * dcinf
        Kqinf = Kcinf * Ko * tan_2

        # pressure at an arbitrary depth
        aq = (Kqo / (Kqinf - Kqo)) * (Ko * sin_2 / sin_3)
        KqD = (Kqo + Kqinf * aq * zbRatio) / (1 + aq * zbRatio)

        # ultimate lateral resistance
        if pyDepth == 0:
            pu = 0.01
        else:
            pu = gamma * pyDepth * KqD * b

        # PySimple1 material formulated with pult as a force, not force/length, multiply by trib. length
        pult = pu * pEleLength

    # ----------------------------------------------------------
    #  define displacement at 50% lateral capacity, y50
    # ----------------------------------------------------------

    # values of y50 depend of the coefficent of subgrade reaction, k, which can be defined in several ways.
    #  for gwtSwitch = 1, k reflects soil above the groundwater table
    #  for gwtSwitch = 2, k reflects soil below the groundwater table
    #  a linear variation of k with depth is defined for kSwitch = 1 after API (1987)
    #  a parabolic variation of k with depth is defined for kSwitch = 2 after Boulanger et al. (2003)

    # API (1987) recommended subgrade modulus for given friction angle, values obtained from figure (approximate)

    ph = [28.8, 29.5, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]

    # subgrade modulus above the water table
    if gwtSwitch == 1:
        k = [10, 23, 45, 61, 80, 100, 120, 140, 160, 182, 215, 250, 275]

    else:
        k = [10, 20, 33, 42, 50, 60, 70, 85, 95, 107, 122, 141, 155]

    dataNum = 13
    for i in range(dataNum):
        if ph[i] <= phiDegree and phiDegree <= ph[i + 1]:
            khat = (k[i + 1] - k[i]) / (ph[i + 1] - ph[i]) * (phiDegree - ph[i]) + k[i]

            # change units from (lb/in^3) to (kN/m^3)
    k_SIunits = khat * 271.45

    # define parabolic distribution of k with depth if desired (i.e. lin_par switch == 2)
    sigV = pyDepth * gamma

    if sigV == 0:
        sigV = 0.01

    if kSwitch == 2:
        # Equation (5-16), Boulanger et al. (2003)
        cSigma = math.pow(50 / sigV, 0.5)
        # Equation (5-15), Boulanger et al. (2003)
        k_SIunits = cSigma * k_SIunits

    # define y50 based on pult and subgrade modulus k

    # based on API (1987) recommendations, p-y curves are described using tanh functions.
    #  tcl does not have the atanh function, so must define this specifically

    #  i.e.  atanh(x) = 1/2*ln((1+x)/(1-x)), |x| < 1

    # when half of full resistance has been mobilized, p(y50)/pult = 0.5
    x = 0.5
    log_1 = math.log((1 + x) / (1 - x))
    atanh_value = 0.5 * log_1

    # need to be careful at ground surface (don't want to divide by zero)
    if pyDepth == 0.0:
        pyDepth = 0.01

    y50 = 0.5 * (pu / A) / (k_SIunits * pyDepth) * atanh_value
    # return pult and y50 parameters
    outResult = []
    outResult.append(pult)
    outResult.append(y50)

    return outResult

op.wipe()

#########################################################################################################################################################################

#########################################################################################################################################################################

# all the units are in SI units N and mm

# ----------------------------------------------------------
#  pile geometry and mesh
# ----------------------------------------------------------

# length of pile head (above ground surface) (m)
L1 = int(input("Enter the top length of pile(m):"))
# length of total pile (m)
P1 = int(input("Enter the Total Pile Length(m):"))
#Embeded length of pile
S1 = P1-L1
# pile diameter
diameter = 1.0
# number of pile elements
nElePile = 100
# pile element length 2
eleSize = L1+S1                         ##
# number of total pile nodes
nNodePile = 1 + nElePile

#Soil Layer List
soil_layer = list()
layer_depth = list()
soil_value = list()
gamma_soil = list()
count_down = list()
layer_count = input("Enter the number of soil layers:")
for i in range(int(layer_count)):
    print("enter the depth", i, "layer:")
    n = input("Depth value:")
    y = input("Type of soil layer as S for sand or C for clay:")
    z = input("Enter the unit wt of soil:")
    if y =='s':
       e = input("Enter angle of friction:")
    if y =='c':
       e = input("cohesion value:")
    gamma_soil.append(float(z))
    soil_value.append(float(e))
    layer_depth.append(int(n))
#    eleSize = eleSize + layer_depth[i]
    soil_layer.append(y)

eleSize = eleSize/nElePile
print ("the size of the elemnt:",eleSize)
#print ('Depth: ',layer_depth[i])
#print('soil layer',soil_layer)

# select pult definition method for p-y curves
# API (default) --> 1
# Brinch Hansen --> 2
puSwitch = 1

# variation in coefficent of subgrade reaction with depth for p-y curves
# API linear variation (default)   --> 1
# modified API parabolic variation --> 2
kSwitch = 1

# effect of ground water on subgrade reaction modulus for p-y curves
# above gwt --> 1
# below gwt --> 2
gwtSwitch = 1

# ----------------------------------------------------------
#  create spring nodes
# ----------------------------------------------------------
# spring nodes created with 3 dim, 3 dof
op.model('basic', '-ndm', 3, '-ndf', 3)
# counter to determine number of embedded nodes
count = 0
count1 = 0
soil_depth = 0
k = 0
# create spring nodes

# 1 to 85 are spring nodes

pile_nodes = dict()
#for j in range(int(layer_count)):
soil_depth = sum(layer_depth) 
if S1 > soil_depth:
   print("The Pile embeded length is lesser than depth of soil layers...soil layer depth ah athigama kudu da venna")
   sys.exit() 
pi_nodes = math.ceil(soil_depth/eleSize)
for i in range(nNodePile):
    zCoord = eleSize * i
    if zCoord <= S1:                ##
        op.node(i + 1, 0.0, 0.0, zCoord)
        op.node(i + 201, 0.0, 0.0, zCoord)
        pile_nodes[i + 1] = (0.0, 0.0, zCoord)
        pile_nodes[i + 201] = (0.0, 0.0, zCoord)
        count = count+1
    else:
        break
#count_down.append(int(i+1))
#    k = k + (i-k)+1

print("Finished creating all spring nodes...",count)
nNodeEmbed = count
for i in range(nNodeEmbed):
    op.fix(i + 1, 1, 1, 1)
    op.fix(i + 201, 0, 1, 1)

print("Finished creating all spring node fixities...",count)
# ----------------------------------------------------------
#  create spring material objects
# ----------------------------------------------------------
s = 0
c = 0
M1 = S1
li_nodes = 0
soil_pile = soil_depth
layer_dep = 0
count_down.reverse()
layer_depth.reverse()
soil_layer.reverse()
gamma_soil.reverse()
soil_value.reverse()
#p-y spring material
for j in range(int(layer_count)):
    soil_pile =soil_pile-layer_depth[j]
    if S1 <= soil_pile:     
      continue
#    if S1 < (soil_pile-layer_depth[j+1]) :
#        continue   
#   M1 = s1 - layer_depth[j] 
#    if S1 <= soil_depth:
    if S1 > soil_pile: 
       layer_dep = S1 - soil_pile
       S1 = S1 - layer_dep
    li_nodes = li_nodes + math.ceil(layer_dep/ eleSize)
    for i in range(s,li_nodes):
               # depth of current py node
        pyDepth = max(0,M1 - eleSize * (i - 1))
        if soil_layer[j] =='s':
#           print("soil")
           pyParam = get_pyParam(pyDepth,gamma_soil[j],soil_value[j], diameter, eleSize, puSwitch,kSwitch,gwtSwitch)
           pult = pyParam[0]
           y50 = pyParam[1]
           op.uniaxialMaterial('PySimple1', i, 2, pult, y50, 0.0)
#            op.uniaxialMaterial('Series','Pysimple1',i)
        if soil_layer[j] == 'c':
            print("clay")
            c +=1
        s += 1
#    li_nodes = li_nodes + math.ceil(layer_dep/ eleSize)    
print("number of clay layer:",c)
print("number of soil layer:",s)
        # procedure to define pult and y50
         #   pyParam = get_softclaynorm(cu)

      #  pult = pyParam[0]
       # y50 = pyParam[1]
      #  op.uniaxialMaterial('PySimple1', i, 2, pult, y50, 0.0)
    #soil_depth = soil_depth - layer_depth[j]
# ----------------------------------------------------------
#  create zero-length elements for springs
# ----------------------------------------------------------

# element at the pile tip (has q-z spring)
#op.element('zeroLength', 1001, 1, 101, '-mat', 1, 101, '-dir', 1, 3)

# remaining elements
#op.element('zeroLength', 1001, 1, 201, '-mat', 1, '-dir', 1)
f = 0
for i in range(0, nNodeEmbed-1):
    op.element('zeroLength', 1000 + i, i+1, 201 + i, '-mat', i, '-dir', 1, 3)
    f =f+1
    print(f)

print("Finished creating all zero-Length elements for springs...")

# ----------------------------------------------------------
#  create pile nodes
# ----------------------------------------------------------

# pile nodes created with 3 dimensions, 6 degrees of freedom
op.model('basic', '-ndm', 3, '-ndf', 6)

# create pile nodes
for i in range(1, nNodePile + 1):
    zCoord = eleSize * i
    op.node(i + 400, 0.0, 0.0, zCoord)

print("Finished creating all pile nodes...")

# create coordinate-transformation object
op.geomTransf('Linear', 1, 0.0, -1.0, 0.0)

# create fixity at pile head (location of loading)
op.fix(400 + nNodePile, 0, 1, 0, 1, 0, 1)

# create fixities for remaining pile nodes
for i in range(401, 400 + nNodePile):
    op.fix(i, 0, 1, 0, 1, 0, 1)

print("Finished creating all pile node fixities...")

# ----------------------------------------------------------
#  define equal dof between pile and spring nodes
# ----------------------------------------------------------

for i in range(1, nNodeEmbed + 1):
    op.equalDOF(400 + i, 200 + i, 1, 3)

print("Finished creating all equal degrees of freedom...")

# ----------------------------------------------------------
#  create elastic pile section
# ----------------------------------------------------------

secTag = 1
E = 25000000.0
A = 0.785
Iz = 0.049
Iy = 0.049

G = 9615385.0
J = 0.098

matTag = 3000
op.section('Elastic', 1, E, A, Iz, Iy, G, J)

# elastic torsional material for combined 3D section
op.uniaxialMaterial('Elastic', 3000, 1e10)

# create combined 3D section
secTag3D = 3
op.section('Aggregator', secTag3D, 3000, 'T', '-section', 1)

#########################################################################################################################################################################

##########################################################################################################################################################################



# elastic pile section
# import elasticPileSection

# ----------------------------------------------------------
#  create pile elements
# ----------------------------------------------------------
op.beamIntegration('Legendre', 1, secTag3D,3)  # we are using gauss-Legendre  integration as it is the default integration scheme used in opensees tcl (check dispBeamColumn)

for i in range(401, 401 + nElePile):
    op.element('dispBeamColumn', i, i, i + 1, 1, 1)

print("Finished creating all pile elements...")
opsl.plot_model()

# ----------------------------------------------------------
#  create recorders
# ----------------------------------------------------------

# record information at specified increments
timeStep = 0.5

# record displacements at pile nodes
op.recorder('Node', '-file', 'pileDisp.out', '-time', '-dT', timeStep, '-nodeRange', 401, 400 + nNodePile, '-dof', 1,2,
            3, 'disp')

# record reaction force in the p-y springs
op.recorder('Node', '-file', 'reaction.out', '-time', '-dT', timeStep, '-nodeRange', 1, nNodePile, '-dof', 1,
            'reaction')

# record element forces in pile elements
op.recorder('Element', '-file', 'pileForce.out', '-time', '-dT', timeStep, '-eleRange', 401, 400 + nElePile,
            'globalForce')

print("Finished creating all recorders...")

# ----------------------------------------------------------
#  create the loading
# ----------------------------------------------------------

op.setTime(10.0)

# apply point load at the uppermost pile node in the x-direction
values = [0.0, 0.0, 1.0, 1.0]
time = [0.0, 10.0, 20.0, 10000.0]

nodeTag = 400 + nNodePile
loadValues = [3500.0, 0.0, 0.0, 0.0, 0.0, 0.0]
op.timeSeries('Path', 1, '-values', *values, '-time', *time, '-factor', 1.0)

op.pattern('Plain', 10, 1)
op.load(nodeTag, *loadValues)

print("Finished creating loading object...")

# ----------------------------------------------------------
#  create the analysis
# ----------------------------------------------------------
op.integrator('LoadControl', 0.05)
op.numberer('RCM')
op.system('SparseGeneral')
op.constraints('Transformation')
op.test('NormDispIncr', 1e-5, 20, 1)
op.algorithm('Newton')
op.analysis('Static')

print("Starting Load Application...")
op.analyze(401)

print("Load Application finished...")
# print("Loading Analysis execution time: [expr $endT-$startT] seconds.")

# op.wipe

op.reactions()
Nodereactions = dict()
Nodedisplacements = dict()
for i in range(401, nodeTag + 1):
    Nodereactions[i] = op.nodeReaction(i)
    Nodedisplacements[i] = op.nodeDisp(i)
print('Node Reactions are: ', Nodereactions)
print('Node Displacements are: ', Nodedisplacements)
