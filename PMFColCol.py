import os
import sys
import subprocess
import numpy as np
import scipy as sp
from scipy import signal
from scipy.fftpack import fft, fftn
from scipy.fftpack import fft, fftfreq, fftshift
import shutil
import datetime
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.optimize import fsolve
from scipy import optimize
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math


def PMFColCol():
    """PMFColCol "Potential of Mean Force Colloid-Colloid" - Generates the Effective interaction between two spherical
		colloidal particles. The current list of forces incorporated are: All energies are in thermal energy units!
		Wvdw(r)      -  van der Waals forces (Everaers, Ejtehadi, Phys Rev E, 67, 041710 (2003).)
		Whs(r)       -  hard sphere repulsion due to lenard-jones interactions (integrated out) (Everaers, Ejtehadi, Phys Rev E, 67, 041710 (2003).)
		Wel(r-2Rc)   - electrostatic interactions pg. 317 Israelachvili
		Wpoly(r-2Rc) - polymer mediated interactions from a transformation of an effective thermodynamic potential
			between two flat plates. The length units are in nanometers and thermal energy.
		
	Notes: 
		(1) The equations below make use of only the statistical segment length ( StatSeg = bseg/sqrt(6) ) as is used in polyFTS.
			Thus, there will need to be an appropriate sqrt(6) correction made to the Col-Col PMF and Force Curves! (typically mult. by sqrt(6))
			
		(2) For the VDW+HS interactions, the force of the last two points are set to zero.
		
		(3) For all forces, the first and last 2 points are extrapolated from quadratic fits to the first/last 3 points.
		
		(4) All energies are that of thermal energy.
		
		(5) Many times the polymer force/potential curves are not started exactly at 0 surface separation and go towards infinity (e.g. brushes)
				as the plates approach. To adjust for this, I have implemented linear extrapolation (could possibly do quadratic, but can be dangerous)
				to extend the curve over the whole range of separations as defined in znew1 (where znew1 came from znew).
		
		(6) The combined force was calculated using the five point stencil formula, this is of fourth order accuracy in the discretization! 
	
	User Input:
		(1) Plate-Plate PMF calculated from PMFFindStandAlone in the pythonPMF.py package
			with dimensionless length in units of Rg
		(2) Hamaker Constant between two colloids, Acc
		(3) Permittivity of the medium, Epsilon
		(4) Charge Valency of the ions, Ze
		(5) Radius of Colloid, Rc (nm)
		(6) StatSeg, bs (nm)



	User needs to set:
		ColRadius in nanometers
		StatSeg Length in nanometers ( bseg/sqrt(6)=lseg )
		NumStatSeg in polymer ( this is the coarse-grained polymer )
		
	OutPuts:
		ColColForce in units J/StatSeg per Thermal Energy
		ColColPMF   in units Thermal Energy
	
	******** Editions and Revisions **********
	Created 2017.05.11 by Nick Sherck
	Revised 2017.09.28 by Nick Sherck - added in Colloid Radius, StatSeg, Prefactor and ColColForce
	Revised 2017.10.23 by Nick Sherck - To incorporate VDW, HS, Electrostatics and Polymer Forces
	Revised 2017.11.12 by Nick Sherck - Added in improved Second Virial Coefficient and Scattering Function
	Revised 2017.11.17 by Nick Sherck - Added in a tscale to scale PMF and Force for different temperatures (Temporary)
	Revised 2017.11.18 by Nick Sherck - Added in the "pressure equation"; pg. 262 in McQuarie AND fixed the Combined Forces Error
	Revised 2017.12.14 by Nick Sherck - Changed the Second Virial Coefficient to be in reduced units of Hard Sphere (2*pi/3/(2*Rc)^3)
	Revised 2019.01-02 by Daniel Chu  - Derivatives switched to 5pt FD, corrected HS repulsion term, made PMF reach MaxVal at 0 distance, added Rg,0/StatSeg flag
	Revised 2019.02.18 by Daniel Chu  - Plate-Plate Potential now linearly extrapolates from IP to remove wall interaction
	Revised 2019.03.17 by Daniel Chu  - Added field-theory characteristic density thing
	"""

    # ********************************* USER SETTINGS ***************************************** #
    # ***************************************************************************************** #

    # INTERACTION RANGE
    # IN UNITS of statistical segment length!!!!!!!!!
    # These are from the surface of the spheres

    # The minimum distance in units of statistical segment length (should be set to 0)
    rngmin = 0

    # The maximum distance in units of statistical segment length (bs/sqrt(6))
    rngmax = 500

    # Maximum cutoff for the PMF between colloids (units of kbT); This is important for LAMMPS (if too
    # large, LAMMPS will not be able to interpolate correctly)
    PMFMaxVal = 50000

    # Temperature Reduction Factor for MD Simulations (Temporary fix)
    ''' TODO: 
		(1) Make this fix more permanent, the final potential is in units of kbT, need to be able to 
				scale this based on a changing reduced temperature!  (the solution should involve chi - 2019.05.20)
		(2) Currently, to handle this, we just set the Temperature to 308K or 35 C for EL and VDW.
				and scale the final combined potential by some factor between tscale, 0< tscale < 1.0
				W/kbT * tscale, where tscale = kbT/epsilon, where epsilon is some characteristic energy
		(3) Check virial coefficient calculations
		(4) Allow file specification and more from command line
	'''
    tscale = 1.0  # keep at 1.0 unless generating different Temperature curves

    ''' POLYMER INDUCED INTERACTIONS '''

    # number of interpolation points that controls numerical accuracy for integration (simpson's) and
    # 	differentiation (5 pt stencil) (needs to be fine for LAMMPS)
    Resolution = 20  # typically 30
    # Adjust Resolution is a parameter for the reduced range output for qualitative plotting in things like excel
    AdjRes = 2
    # Colloid Radius
    ColRadius = 13.77  # units (nm)
    Rc = ColRadius
    # StatSeg Length/sqrt(6) lseg = bseg/sqrt(6)
    StatSeg = 0.177  # statistical segment length (nm)
    # Number of Statistical Segments
    NumStatSeg = 34  # number of statistical segments in the polymer
    # Characteristic density from field theory
    rho = 0.0299
    # Flat-Plate Potential File Name (needs to be in units of Rg) &
    # Only should have two columns.
    FPPMFName = "ChiNeg3.txt"
    # Flag Polymer turns off the polymer in the final potential/force (1 = on ; 0 = off)
    FlagPolymer = 1.0
    # FlagStatSeg changes the data input between Rg,o and statistical segment lengths (1 = Rg,o, & 0 = StatSeg)
    FlagStatSeg = 0.0
    ''' ELECTROSTATICS - Set Surface Charge to Zero if Negligible '''

    # Temperature
    T = 313  # (K)
    print("Temperature")
    print(T)
    # Surface Charge of Colloid
    SurfCharge = 0.050  # (Volts)
    # Ion Concentration (for electrostatics)
    IonConc = 0.010  # Molar (mols/Liter)
    # Number of Ions (the valency is the next option)
    IonNum = 2.0  # charge
    # Ionic Valency (1:1 ions = 1; divalent = 2; rms of valency if different?)
    IonVal = 1.0
    # Relative Permittivity of Medium
    RelPerm = 78.4  # (unitless, 78.4 for Water)
    # Flag Electrostatics Turns Electrostatics on and Off (1 = on ; 0 = off)
    FlagEl = 1.0

    ''' HARD-SPHERE REPULSION & VDW ATTRACTION - Set AccHS to Zero if HS Negligible or SET AccVDW to Zero to remove VDW '''

    # Fundamental bead size in LJ for colloid (in hard-repulsion); Defaults to StatSeg
    Sigma = StatSeg  # (nm)
    # Hamaker Constant VDW in Joules
    AccVDW = 1 * 3.3 * 10 ** -22  # J
    # Hamaker Constant HS (Typically the same as VDW, just defined two to shutoff one or the other)
    AccHS = 1 * 3.3 * 10 ** -22  # J

    # ************************************* CONSTANTS ***************************************** #
    # ***************************************************************************************** #
    Perm = 8.854 * 10 ** -12  # F/m = J/V**2
    Echrg = 1.602 * 10 ** -19  # J/V
    kb = 1.381 * 10 ** -23  # J/molecule/K
    Nav = 6.022 * 10 ** 23  # molecules/mole
    Debye = ((Perm * RelPerm * kb * T) ** 0.5 / (
                IonNum * IonVal ** 2 * Nav * 1000 * Echrg ** 2 * IonConc) ** 0.5) * 10 ** 9  # Debye Length (nm)
    # "interaction constant" eqn 14.52 Israel. Units of Thermal Energy per nm
    x = float((Echrg / kb / T) * IonVal * SurfCharge / 4)
    preZ = FlagEl * 64 * math.pi * Perm * RelPerm * (kb * T / (Echrg)) ** 2 * math.tanh(x) ** 2 / (kb * T) * (1E-9)
    print("Debye Length")
    print(Debye)
    print("Interaction Constant - Electrostatics")
    print(preZ)

    # ********************************************** IMPORT FLAT PLATE POTENTIAL (POLYMER) ******************************************** #
    # ************************************************************************************ ******************************************** #
    # ********************************************************************************************************************************* #
    [zrange, potential] = np.hsplit(np.genfromtxt(FPPMFName, dtype=None, comments="#"), 2)
    zrange = np.asarray(zrange).squeeze()
    potential = np.asarray(potential).squeeze()
    # newDat = np.column_stack((zrange, potential))  # Generate the ouput effective colloid-colloid VDW force file
    # np.savetxt("ChiNeg3.csv", newDat,delimiter=',')  # Save to text file

    # convert zrange to units of StatSegLength ( b/sqrt(6) ); input data in Rg,o units
    zrange = [i * (FlagStatSeg * NumStatSeg ** 0.5 + (1 - FlagStatSeg)) for i in zrange]

    # ********************************* FLAT PLATE DATA PROCESSING (POLYMER) ****************************** #
    # ***************************************************************************************************** #

    zrlen = len(zrange)
    zrangemin = min(zrange)
    zrangemax = max(zrange)

    # NOTE: znew in units of statistical segment length
    znew = np.linspace(zrangemin, zrangemax, ((zrlen - 1) * Resolution))  # sets the number of pts in ColColPot
    # print("znew")
    # print(znew)

    smoothpot2 = interpolate.splrep(zrange, potential)  # interpolation for differentiation
    concavity = interpolate.splev(zrange, smoothpot2, der=2)  # second derivative of PP potential
    # does concavity change at i, closer to wall than one Rg,0
    signswitch = [concavity[i] * concavity[i - 1] < 0 and zrange[i] < NumStatSeg ** 0.5 for i in range(1, zrlen)]
    if True in signswitch:
        signswitch = signswitch.index(True) + 1  # find inflection point
        ExtrapSlope = interpolate.splev(zrange, smoothpot2, der=1)[signswitch]  # derivative at inflection point
        for i in range(0, signswitch):  # replace potential values with linear extrapolation
            potential[i] = (zrange[i] - zrange[signswitch]) * ExtrapSlope + potential[signswitch]

        zeroPot = potential[signswitch] - zrange[signswitch] * ExtrapSlope  # extrapolated potential at 0 separation
    else:
        zeroPot = potential[0] - zrange[0] * interpolate.splev(zrange[0], smoothpot2, der=1)
    smoothpotential = interp1d(zrange, potential, kind='cubic')  # Spline fitting to the new data

    # the spacing resolution
    zRes = znew[1] - znew[0]
    print('Spatial Resolution (Stat. Seg.)')
    print(zRes)
    print("Spatial Resolution(nm)")
    print(zRes * StatSeg)

    zlng = len(znew)
    ColColPolymerPot = []
    prefactor = ColRadius * math.pi / StatSeg * (StatSeg ** 3 / rho)  # Prefactor that multiplies the integrated PMF

    # Colloid-Colloid Polymer Force Calculation in Units of (J/StatSegLength/kbT)
    ColColPolymerForce = [prefactor * smoothpotential(i) for i in znew]

    # Colloid-Colloid Polymer Potential Calculation
    # Trapezoidal rule for integration
    ColColPolymerPot = [prefactor * integrate.trapz(smoothpotential(znew[i:zlng]), znew[i:zlng]) for i in
                        range(0, zlng)]

    # Add in ColRadius (center to center potential)
    znewsurface = [i for i in znew]  # for plotting purposes below
    znew = [i * StatSeg + 2 * ColRadius for i in znew]  # znew has units of (nm) ; znewsurface has units of (StatSeg)

    # Add in ColRadius
    zrange = [i + 2 * ColRadius / StatSeg for i in zrange]

    PlatePlatePot = np.column_stack(
        (zrange, potential))  # Generate Plate-Plate Polymer Potential in new Units for plotting purposes
    ColColPolymerForce = np.column_stack(
        (znew, ColColPolymerForce))  # Generate the ouput effective colloid-colloid Polymer force file
    ColColPolymerPotential = np.column_stack(
        (znew, ColColPolymerPot))  # Generate the output effective colloid-colloid Polymer potential file
    # need the zlng-1 due to the fact that simpsons rule removes one pt

    # ********************************************** END POLYMER FORCE/POTENTIAL ****************************************************** #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    # Redefine znew and znewsurface in units of nm that are commensurate with the znew above
    # znew1 center to center (nm)
    # znewsurface1 from surface to surface (nm)

    print("zrange minimum")
    print(zrangemin)
    min1 = (zrangemin * StatSeg)  # The mimimum distance in nm of the polymer PMF
    chk1 = int(min1 / zRes / StatSeg)  # number of whole steps between 0 and zrangemin
    rngmin = min1 - chk1 * zRes * StatSeg  # Guarantees that the profiles will align and avoids having a zero in the range (bad for division)
    print("Number of Steps From Min. Distance to Polymer PMF Starting Distance")
    print(chk1)
    print("Polymer Minimum Distance (nm)")
    print(min1)
    print("PMF/Force Starting Distance (nm)")
    print(rngmin)

    cntStart = rngmin  # the offset to match with polymer range
    cnt = 0
    i = 0
    znew1 = []
    znewsurface1 = []
    while cnt < rngmax * StatSeg:
        cnt = i * zRes * StatSeg + cntStart  # zRes set by the polymer input file (nm)
        temp = cnt + 2 * Rc  # the center to center distance (nm)
        znew1.append(temp)
        znewsurface1.append(cnt)
        i = i + 1

    zlng = len(znew1)

    # ********************************************** START VAN DER WAALS FORCE/POTENTIAL*********************************************** #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    # Both the VDW and the HS interactions and forces normalized by thermal energy
    ColColVDWPot = []
    ColColVDWForce = []
    # znew is now in units of nm
    for i in znew1:

        chk = 2 * Rc
        if i <= chk:  # Check value is 2*Rc
            tempAtt = 0
            tempHS = 10000 * np.heaviside(AccHS, 0)  # turn off HS if AccHS = 0
        else:
            tempAtt = (-1 * AccVDW / 6 / kb / T) * (
                        (2 * Rc ** 2 / (i ** 2 - 4 * Rc ** 2)) + (2 * Rc ** 2 / i ** 2) + math.log(
                    (i ** 2 - 4 * Rc ** 2) / (i ** 2)))
            tempHS = (AccHS / kb / T * Sigma ** 6 / 37800 / i) * (
                        ((i ** 2 - 14 * i * Rc + 54 * Rc ** 2) / (i - 2 * Rc) ** 7) + (
                            (i ** 2 + 14 * i * Rc + 54 * Rc ** 2) / (i + 2 * Rc) ** 7) - 2 * (
                                    (i ** 2 - 30 * Rc ** 2) / (i ** 7)))
        tempComb = tempAtt + tempHS
        ColColVDWPot.append(tempComb)

    # Take the numerical derivative the fourth order accurate, 5-pt stencil
    h = znew1[1] - znew1[0]
    for i in range(0, zlng):
        if 2 <= i < zlng - 2:  # use central differences for interior points
            DerVal = (ColColVDWPot[i - 2] - ColColVDWPot[i - 1] * 8 + ColColVDWPot[i + 1] * 8 - ColColVDWPot[
                i + 2]) / 12 / h
        elif i < 2:  # use forward differences for first two points
            DerVal = (ColColVDWPot[i] * (-25) + ColColVDWPot[i + 1] * 48 - ColColVDWPot[i + 2] * 36 + ColColVDWPot[
                i + 3] * 16 - ColColVDWPot[i + 4] * 3) / 12 / h
        else:  # use backwards differences for last two points
            DerVal = (ColColVDWPot[i] * 25 - ColColVDWPot[i - 1] * 48 + ColColVDWPot[i - 2] * 36 - ColColVDWPot[
                i - 3] * 16 + ColColVDWPot[i - 4] * 3) / 12 / h
        ColColVDWForce.append(-1. * DerVal)

    # ********** VISUAL VDW ***********************

    plt.figure()
    plt.plot(znew1, ColColVDWPot, 'b-', lw=3, label="Colloid-Colloid VDW+HS PMF")
    plt.plot(znew1, ColColVDWForce, 'r-', lw=3, label="Colloid-Colloid VDW+HS Force")
    plt.xlabel("h (nm)")
    plt.ylabel("PMF")
    plt.title("Colloid-Colloid VDW+HS PMF & Force")
    plt.legend()
    plt.savefig('ColColVDWHS.pdf')
    plt.show()
    plt.close()

    ColColVDWForce = np.column_stack(
        (znew1, ColColVDWForce))  # Generate the ouput effective colloid-colloid VDW force file
    ColColVDWPotential = np.column_stack(
        (znew1, ColColVDWPot))  # Generate the output effective colloid-colloid VDW potential file

    np.savetxt("ColColVDWForce.txt", ColColVDWForce)  # Save to text file
    np.savetxt("ColColVDWPMF.txt", ColColVDWPotential)  # Save to a text file

    # ********************************************** END VDW + HS FORCE/POTENTIAL ***************************************************** #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    # ********************************************** START ELECTROSTATIC FORCE/POTENTIAL ********************************************** #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    ColColELForce = []
    ColColELPot = []

    for i in znewsurface1:  # use znewsurface1 (nm) since the electrostatic force is from the surface of the sphere

        # preZ is in units of thermal energy per nm
        temp = Rc / 2 * preZ * math.exp(-1 * i / Debye)
        # Negative 1 out front, because the force is the negative of the derivative of the potential energy
        temp1 = -1 * (-1 * Rc / 2 * preZ / Debye * math.exp(-1 * i / Debye))

        ColColELPot.append(temp)
        ColColELForce.append(temp1)

    # ********** VISUAL EL ***********************

    plt.figure()
    plt.plot(znewsurface1, ColColELPot, 'b-', lw=3, label="Colloid-Colloid EL PMF")
    plt.plot(znewsurface1, ColColELForce, 'r-', lw=3, label="Colloid-Colloid EL Force")
    plt.xlabel("surface separation (nm)")
    plt.ylabel("PMF")
    plt.title("Colloid-Colloid Electrostatic PMF & Force")
    plt.legend()
    plt.savefig('ColColEL.pdf')
    plt.show()
    plt.close()

    ColColELForce = np.column_stack(
        (znewsurface1, ColColELForce))  # Generate the ouput effective colloid-colloid VDW force file
    ColColELPotential = np.column_stack(
        (znewsurface1, ColColELPot))  # Generate the output effective colloid-colloid VDW potential file

    np.savetxt("ColColELForce.txt", ColColELForce)  # Save to text file
    np.savetxt("ColColELPMF.txt", ColColELPotential)  # Save to a text file

    # ********************************************** END ELECTROSTATIC FORCE/POTENTIAL ************************************************ #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    # ************************************************ COMBINING FORCE/POTENTIAL ****************************************************** #
    # ********************************************************************************************************************************* #
    # ********************************************************************************************************************************* #

    cnt = 0
    ColColCombPot = []
    ColColCombForce = []
    ColColPolymerPMFExt = []

    plng = len(ColColPolymerPotential)

    # Prepares Quadratic extrapolation of the polymer force curve!
    MaxPeakPolyCoef = np.polyfit((znew[0:3]), ColColPolymerPotential[0:3, 1], 2, full=True)  # fit first-order (linear)
    coef1 = MaxPeakPolyCoef[0]  # polymer coeficients

    # print "znew and ColColPolymerPot"
    # print znew[0:3], ColColPolymerPot[0:3]
    # print "Polynomial Coefficients for Quadratic Extrapolation of Polymer PMF"
    # print coef1

    for i in znewsurface1:
        if cnt < chk1:  # This is from above when redefining the znew1 values
            # Quadratic extrapolation of the last four points on the polymer force.
            rval = i + 2 * Rc
            temp0 = coef1[0] * rval ** 2 + coef1[1] * rval + coef1[2]

        elif cnt >= chk1 and cnt < plng - 1 + chk1:  # pull values from polymer force
            temp0 = ColColPolymerPotential[cnt - chk1][1]
        else:
            temp0 = 0

        if FlagPolymer == 0:  # Shuts off the polymer interaction
            temp0 = temp0 * 0

        temp1 = ColColVDWPotential[cnt][1]
        temp2 = ColColELPotential[cnt][1]

        temptotPot = temp0 + temp1 + temp2

        # NOTE that below, tscale is a temporary fix to rescale the output potential
        ColColCombPot.append(temptotPot / tscale)
        ColColPolymerPMFExt.append(temp0)  # Extrapolated polymer potential (for visual reference)

        cnt = cnt + 1

    # FORCES
    # Calculate the forces w/ five point stencil formula (fourth order accurate)
    cnt = 0
    lngmax = len(ColColCombPot)

    # Convert to nm to 2*Rc Units (center to center distance)
    znew2 = []
    znew2Rg = []
    for i in znew1:
        znew2.append(i / 2 / Rc)
        znew2Rg.append((i - 2 * Rc) / 2 / math.sqrt(NumStatSeg) / StatSeg)

    dx = znew2[1] - znew2[0]

    for i in znew2:  # Take the derivative

        if cnt >= 2 and cnt < zlng - 2:
            DerVal = (-1 * ColColCombPot[cnt + 2] + 8 * ColColCombPot[cnt + 1] - 8 * ColColCombPot[cnt - 1] +
                      ColColCombPot[cnt - 2]) / 12 / dx
        elif cnt < 2:
            DerVal = (ColColCombPot[cnt] * (-25) + ColColCombPot[cnt + 1] * 48 - ColColCombPot[cnt + 2] * 36 +
                      ColColCombPot[cnt + 3] * 16 - ColColCombPot[cnt + 4] * 3) / 12 / dx
        else:
            DerVal = (ColColCombPot[cnt] * (25) - ColColCombPot[cnt - 1] * 48 + ColColCombPot[cnt - 2] * 36 -
                      ColColCombPot[cnt - 3] * 16 + ColColCombPot[cnt - 4] * 3) / 12 / dx

        cnt = cnt + 1
        DerVal = -1 * DerVal

        ColColCombForce.append(DerVal)

    # ********** VISUAL COMBINED ***********************

    plt.figure()
    plt.plot(znew1, ColColCombPot, 'b-', lw=3, label="Colloid-Colloid PMF")
    plt.plot(znew1, ColColCombForce, 'r-', lw=3, label="Colloid-Colloid Force")
    plt.xlabel("h (nm)")
    plt.ylabel("PMF")
    plt.title("Colloid-Colloid Total PMF & Force")
    plt.legend()
    plt.savefig('ColColComb.pdf')
    plt.show()
    plt.close()

    # Indexing for LAMMPS INPUT
    cnt = 1
    index = []
    for i in znew1:
        index.append(cnt)
        cnt = cnt + 1

    ColColComb = np.column_stack((index, znew2, ColColCombPot, ColColCombForce))  # combine all for input into lammps

    # evaluate linear extrapolations of potential to origin compared to PMFMaxVal
    linextrap = [ColColComb[i][2] + ColColComb[i][3] * ColColComb[i][1] - PMFMaxVal for i in range(0, len(ColColComb))]
    log1 = abs(np.diff(np.sign(linextrap))) == 2  # see where the extrapolations cross PMFMaxVal
    linextrap = [linextrap[i] for i in range(1, len(ColColComb))]  # redefine extrapolations to match len(log1)
    log2 = [abs(i) < PMFMaxVal for i in linextrap]  # make sure that we don't include initial oscillations
    strt = [log1[i] and log2[i] for i in range(0, len(log2))]  # logical array of zero crossings w/o wild oscillations
    if True in strt:  # if there is a satisfactory crossing, perform extrapolation
        strt = strt.index(True) + 1  # find index to extrapolate from
        for i in range(0, strt):  # complete extrapolation of potential with constant force
            ColColComb[i][2] = ColColComb[strt][2] + ColColComb[strt][3] * (ColColComb[strt][1] - ColColComb[i][1])
            ColColComb[i][3] = ColColComb[strt][3]

    # Generate tabulated LAMMPS input for pair potential and force
    f = open("PMFColCol.Table", "w")

    zmin = min(znew1) / 2 / Rc
    zmax = max(znew1) / 2 / Rc

    cnt = 0
    for i in ColColComb:
        if i[2] <= PMFMaxVal:
            if cnt == 0:
                zmin = i[1]
                f.write('N ' + str(zlng) + ' R ' + str(zmin) + ' ' + str(zmax) + '\n')
                f.write('\n')
            cnt = cnt + 1
            f.write('%6i %.10e %.10e %.10e \n' % (cnt, i[1], i[2], i[3]))

    f.close()

    plt.figure()
    plt.plot(znew2Rg, [i[2] for i in ColColComb], 'r-', lw=2, label="Combined")
    plt.plot([i / 2 / math.sqrt(NumStatSeg) for i in znewsurface], ColColPolymerPot, 'b-', lw=1, label="Polymer")
    plt.plot(znew2Rg, ColColVDWPot, 'c-', lw=1, label="VDW")
    plt.plot(znew2Rg, ColColELPot, 'g-', lw=1, label="EL")
    plt.xlabel("r' = (r - 2Rc)/Rg,o")
    plt.ylabel("PMF")
    plt.ylim((-1.5 * abs(min([i[2] for i in ColColComb]))), 1.5 * abs(min([i[2] for i in ColColComb])))
    plt.xlim((0, 4))
    plt.title("Colloid-Colloid Potentials")
    plt.legend()
    plt.savefig('AllPot.pdf')
    plt.show()
    plt.close()

    # Reduced Points because Scott Cannot Get his stuff together.
    ColColPolymerPMFExtRed = ColColPolymerPMFExt[0::Resolution * AdjRes]
    znew2Rg = znew2Rg[0::Resolution * AdjRes]

    ColColCombForce = np.column_stack(
        (znew1, ColColCombForce))  # Generate the ouput effective colloid-colloid VDW force file
    ColColCombPotential = np.column_stack(
        (znew1, ColColCombPot))  # Generate the output effective colloid-colloid VDW potential file
    ColColPolymerPMFExt = np.column_stack(
        (znew1, ColColPolymerPMFExt))  # Generates effective output for the extrapolated polymer potential
    ColColPolymerPMFExtRed = np.column_stack((znew2Rg, ColColPolymerPMFExtRed))

    np.savetxt("ColColCombForce.csv", ColColCombForce, delimiter=',')  # Save to text file
    np.savetxt("ColColCombPMF.csv", ColColCombPotential, delimiter=',')  # Save to a text file
    np.savetxt("ColColPolymerPMFExt.txt", ColColPolymerPMFExt)
    np.savetxt("ColColPolymerPMFExtRed.txt", ColColPolymerPMFExtRed)

    # ************ SECOND VIRIAL COEFFICIENT & PRESSURE EQN & SCATTERING FUNCTION ****************************** #
    # ***********************************************************************************************************#
    '''
		Defined as in McQuarie pg. 228: Beta2 = 2/3*pi*(2*Rc)^3 - 2*pi int//dr [e^(-u(r)/kbT) -1]*r^2
			- The first term is the "Hard-Sphere Term" and the second is the PMF contribution
			
		NOTE: B2 is dimensionless, in units of (2*Rc)^3!
	'''

    B2HS = 2 * math.pi / 3
    print("Hard Sphere Contribution Units [(2*Rc)^3]")
    print(B2HS)

    # Construct the integrand
    cnt = 0
    temp0 = 0
    temp01 = 0
    temp1 = []
    temp2 = []  # for the pressure equation
    for i in znew2:
        temp0 = (math.exp(-1 * ColColComb[cnt][2]) - 1) * i ** 2
        temp1.append(temp0)
        temp01 = 4 * math.pi / 6 * -1 * ColColComb[cnt][3] * math.exp(-1 * ColColComb[cnt][2]) * i ** 3
        temp2.append(temp01)
        cnt = cnt + 1

    B2Correction = 2 * math.pi * integrate.simps(temp1, znew2)
    print("B2 Correction Term Units [(2*Rc)^3]")
    print(B2Correction)
    B2 = B2HS - B2Correction
    PeqnCoef = integrate.simps(temp2, znew2)
    print("Second Virial B2:B2HS [2*pi/3 (2*Rc)^3]")
    B2vB2HS = B2 / B2HS
    print(B2vB2HS)
    print("Second Virial in cm^3 per mol *10-8 Units")
    print(B2 * 10 ** -8 * (2 * ColRadius * 10 ** -7) ** 3 * 6.022 * 10 ** 23)
    print("Pressure Equation Integral in Units [kbT*(2*Rc)^3]")
    print(PeqnCoef)

    virFile = open("VirialData.txt", "w")
    virFile.write("B2 Correction Term Units [ (2*Rc)^3 ] /n")
    virFile.write("%2.5e /n" % B2Correction)
    virFile.write("/n")
    virFile.write("Reduced Second Virial Coefficient B2:B2HS [ 2*pi/3 (2*Rc)^3 ] /n")
    virFile.write("%2.5e /n" % B2vB2HS)
    virFile.write("/n Second Virial in cm^3 per mol*10EE-8 Units /n")
    virFile.write("/n Pressure Equation Integral in Units [kbT*(2*Rc)^3] /n")
    virFile.write("%2.5e /n" % PeqnCoef)


if __name__ == "__main__":  # Code that runs the program if called from the command line
    PMFColCol()
