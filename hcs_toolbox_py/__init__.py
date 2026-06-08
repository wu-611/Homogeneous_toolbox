"""
HCS Toolbox for Python — Homogeneous Control Systems Toolbox (ver 0.2)

Python port of the MATLAB HCS_Toolbox_ver02 by Andrey Polyakov.
Provides functions for upgrading linear controllers (LPC, LPIC) to
homogeneous ones (HPC, HPIC, FHPC, FHPIC).

Core functions:
  - hnorm       : canonical homogeneous norm
  - hproj       : homogeneous projection onto unit sphere
  - hphi        : homogeneous homeomorphism (forward & inverse)
  - hadd        : addition in homogeneous Euclidean space
  - hdot        : scalar multiplication in homogeneous Euclidean space
  - hinner      : inner product in homogeneous Euclidean space
  - lp_norm     : Lp norm of a signal over a time interval
  - block_con   : block controllability canonical form
  - trans_con   : orthogonal transformation for block decomposition
  - ZOH         : zero-order hold discretization
  - output_form : reduce dynamics of the output via feedback
  - lpc2hpc     : upgrade linear proportional control → HPC
  - lpic2hpic   : upgrade linear PI control → HPIC
  - e_hpc       : evaluate HPC control law
  - e_hpic      : evaluate HPIC control law
  - e_fhpc      : evaluate fixed-time HPC control law
  - e_fhpic     : evaluate fixed-time HPIC control law
  - hpc_design  : direct HPC design (no linear prototype)
  - hpic_design : direct HPIC design
  - fhpc_design : direct FHPC design
  - fhpic_design: direct FHPIC design
  - lo2ho       : upgrade linear observer → homogeneous observer
  - e_ho        : evaluate HO (explicit Euler discretization)
  - si_ho       : evaluate HO (semi-implicit discretization)
"""

from .hnorm import hnorm
from .hproj import hproj
from .hphi import hphi, hphi_inv
from .hadd import hadd
from .hdot import hdot
from .hinner import hinner
from .lp_norm import lp_norm
from .block_con import block_con
from .trans_con import trans_con
from .ZOH import ZOH
from .output_form import output_form
from .lpc2hpc import lpc2hpc
from .lpic2hpic import lpic2hpic
from .e_hpc import e_hpc
from .e_hpic import e_hpic
from .e_fhpc import e_fhpc
from .e_fhpic import e_fhpic
from .hpc_design import hpc_design
from .hpic_design import hpic_design
from .fhpc_design import fhpc_design
from .fhpic_design import fhpic_design
from .lo2ho import lo2ho
from .e_ho import e_ho
from .si_ho import si_ho
