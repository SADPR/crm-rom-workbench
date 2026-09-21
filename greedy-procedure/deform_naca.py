import numba as nb
import numpy as np
from joblib import Parallel, delayed
import os
import time
import matplotlib.pyplot as plt

#@nb.njit(parallel=False)
#@nb.njit(parallel=True)
@nb.njit((nb.float64[:, :], nb.float64[:, :], nb.int32), parallel=True)
def distanceMatrixPower(a, b, p):
  res = np.empty((a.shape[0], b.shape[0]), dtype=a.dtype)
  for i in nb.prange(a.shape[0]):
  #for i in range(a.shape[0]):
    for j in range(b.shape[0]):
      #res[i,j] = np.sum(np.abs(a[i,:] - b[j,:])**p)
      s = 0.0
      for k in range(b.shape[1]):
        s = s + abs(a[i, k] - b[j, k])**p
      res[i,j] = s
  return res

# wrapper because numba does not allow np.float_power
def distanceMatrix(a, b, p):
  return np.float_power(distanceMatrixPower(a, b, p), 1.0/float(p))


################################################################################
def load_mesh(fno, ffluid = None, fstick = None):
  """ 
  Load pieces of the top file
  :param fno: filename with the nodal coordinates; the first column with
              the node numbers is ignored
  :param ffluid: filename with the fluid set; element numbers and types
                 are ignored
  :param fstick: filename with the wall elements; element numbers and types
                 are ignored
  :return: nodal coordinates and fluid and wall elements as numpy arrays
  """ 
  no_ = np.loadtxt(fno, dtype = np.float64)
  no = no_[:,1:]
  if ffluid is not None:
    fluid_ = np.loadtxt(ffluid, dtype = np.int32)
    fluid = fluid_[:,2:]
  else:
    fluid = None
  if fstick is not None:
    stick_ = np.loadtxt(fstick, dtype = np.int32)
    stick = stick_[:,2:]
  else:
    stick = None
  return no, fluid, stick

################################################################################
def naca_point_and_tangent(x, P, M, T, section = 'lower'):
  """ 
  Generate surface points and tangent vectors of the four digit NACA airfoil
  :param x: array of parameters within a [0,1] range
  :param P: position of the maximum camber
  :param M: maximum camber
  :param T: thickness
  :param section: 'upper', 'lower', 'arc' generates the three pieces of the
                  airfoil 
  :return: arrays of x coordinates, y coordinates, and their respective 
           derivatives w.r.t. the parameter
  """ 

  if section == 'arc':
    # Definition of arc points from NACA3.py/NACA4.py
    #RT change
    #n      =   100
    n      =   1000
    beta   =   np.linspace(0, np.pi, n)
    xx = 0.5*(1.0-np.cos(beta))
    xu, yu, dxu_dx, dyu_dx = naca_point_and_tangent(xx, P, M, T,
                                                   section = 'upper')
    xl, yl, dxl_dx, dyl_dx = naca_point_and_tangent(xx, P, M, T,
                                                   section = 'lower')

    UpperNotFliped  = np.column_stack((xu[0:n],yu[0:n]))#,zu[0:n]))
    Upper           = np.flipud(UpperNotFliped)
    Lower           = np.flipud(np.column_stack((xl[0:n],yl[0:n])))#,zl[0:n])))
    BothUL          = np.row_stack((UpperNotFliped,Lower))

    # Circle center - this is not pretty but basically computes the center so
    # the the circle goes through the end points and 
    # the airfoil lines are tangent
    A = np.array([ 
       [ BothUL[n-1,1] - BothUL[n-2,1], BothUL[n,1] - BothUL[n+1,1]],
       [-(BothUL[n-1,0] - BothUL[n-2,0]), -(BothUL[n,0] - BothUL[n+1,0])]],
        dtype=np.float64)
    B = np.array( [ BothUL[n,0] - BothUL[n-1,0], BothUL[n,1] - BothUL[n-1,1] ],
                  dtype=np.float64)
    cc = np.linalg.solve(A,B)
    xc = np.array([BothUL[n-1,0],BothUL[n-1,1]], dtype=np.float64) + cc[1] * \
     np.array([ BothUL[n-1,1]-BothUL[n-2,1], -(BothUL[n-1,0]-BothUL[n-2,0]) ],
               dtype=np.float64)
    #print(xc)

    phi1 = np.arctan2(-(xc[1]-BothUL[n-1,1]),-(xc[0]-BothUL[n-1,0]))
    #print(phi1)
    phi2 = np.arctan2(-(xc[1]-BothUL[n,1]),-(xc[0]-BothUL[n,0]))
    #print(phi2)
    r = np.sqrt( (xc[0]-BothUL[n,0])*(xc[0]-BothUL[n,0]) +
                (xc[1]-BothUL[n,1])*(xc[1]-BothUL[n,1]) )
    phi = np.linspace(phi1, phi2, len(x))
    #print(r)
    xs  = xc[0] + r * np.cos(phi)
    ys  = xc[1] + r * np.sin(phi)
    dxs_dx = -r * np.sin(phi)
    dys_dx = r * np.cos(phi)
    circle = []
    return xs, ys, dxs_dx, dys_dx, xc[0], xc[1], phi1, phi2
 
  yc = np.zeros_like(x) 
  dyc_dx = np.zeros_like(x) 
  d2yc_dx2 = np.zeros_like(x) 
  # Camber
  ix = np.nonzero(x < P)
  tmp = M / P**2
  yc[ix] = tmp * (2.0 * P * x[ix] - x[ix]**2)
  dyc_dx[ix] = 2.0 * tmp * (P - x[ix])
  d2yc_dx2[ix] = - 2.0 * tmp
  ix = np.nonzero(x >= P)
  tmp = M / (1 - P)**2 
  yc[ix] = tmp * (1 - (2 * P) + (2 * P * x[ix]) - (x[ix]**2))
  dyc_dx[ix] = 2.0 * tmp * (P - x[ix])
  d2yc_dx2[ix] = - 2.0 * tmp
  
  theta = np.arctan(dyc_dx)

  dtheta_dx = 1.0 / (1.0 + dyc_dx ** 2) * d2yc_dx2
  
  # Thickness Distribution
  a0     =   0.2969
  a1     =  -0.126
  a2     =  -0.3516
  a3     =   0.2843
  #a4     =  -0.1032 # −0.1036
  a4     =  -0.1015
  
  yt = 5.0 * T * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)
  # Adding a regularization parameter at x = 0 
  dyt_dx =  5.0 * T * (a0 * 0.5 /np.sqrt(x + 1e-16)  + \
                       a1 + x * (2.0 * a2  + x * (3.0 * a3 + 4.0 * a4 * x)))
  if section == 'lower':
    # Lower surface
    xs = x + yt * np.sin(theta)
    ys = yc - yt * np.cos(theta)
    dxs_dx = 1.0 + dyt_dx * np.sin(theta) + yt * np.cos(theta) * dtheta_dx
    dys_dx = dyc_dx - dyt_dx * np.cos(theta) + yt * np.sin(theta) * dtheta_dx
  else:
    # Upper surface
    xs = x - yt * np.sin(theta)
    ys = yc + yt * np.cos(theta)
    dxs_dx = 1.0 - dyt_dx * np.sin(theta) - yt * np.cos(theta) * dtheta_dx
    dys_dx = dyc_dx + dyt_dx * np.cos(theta) - yt * np.sin(theta) * dtheta_dx

  return xs, ys, dxs_dx, dys_dx

################################################################################
#@njit
def eval_dist(p, xu, yu, dxu_dx, dyu_dx, flip_normal = False):
  """ 
  Evaluate distance of a point from a set of lines given by arrays of point 
  coordinates on a curve and tangential vectors of the curve and find which 
  line comes closest
  :param p: array of length 2 (x and y coordinate) describing a point
  :param xu: array of x-coordinates
  :param yu: array of y-coordinates
  :param dxu_dx: array of derivatives of x-coordinates w.r.t. to the curve
                 parameter
  :param dyu_dx: array of derivatives of y-coordinates w.r.t. to the curve
                 parameter
  :param flip_normal: flip the normal convention
  :return: distance to the closest line, distance to the respective surface
           point, and index of this point (generating the closest line)
  """ 
  
  ap = np.vstack((xu, yu)).T - p
  n = np.vstack((dyu_dx, -dxu_dx))
  #norm = np.linalg.norm(n, axis = 0)
  norm = np.sqrt(dyu_dx * dyu_dx + dxu_dx * dxu_dx)
  n = (n / norm).T
  if flip_normal:
    n = -n
  #d = np.linalg.norm(ap - (n.T * np.sum(ap * n, axis = 1)).T, axis = 1)
  t = np.sum(ap * n, axis = 1)
  v = ap - (n.T * t).T
  d = np.sqrt(np.sum(v * v, axis = 1))
  # Eliminate points whose lines are shot
  d[ np.nonzero(t>0.01) ] = 1e6 * (p[0] + 100.0)
  ix = np.argmin(d)
  return d[ix], np.linalg.norm(np.array([xu[ix],yu[ix]])-p), ix

################################################################################
def eval_local_coords(p, xu, yu, dxu_dx, dyu_dx, ns):
  """ 
  Evaluate coordinates of p - [xu,yu] with respect to
  a system given by a tangential and normal vector to the curve
  :param p: array of length 2 (x and y coordinate) describing a point
  :param xu: array of x-coordinates
  :param yu: array of y-coordinates
  :param dxu_dx: array of derivatives of x-coordinates w.r.t. to the curve
                 parameter
  :param dyu_dx: array of derivatives of y-coordinates w.r.t. to the curve
                 parameter
  :param ns: normal sign
  :return: coordinates of p in the local system
  """ 
  
  ap = np.vstack((xu, yu)).T - p
  tau = np.vstack((dxu_dx, dyu_dx)) * ns
  norm = np.sqrt(dyu_dx * dyu_dx + dxu_dx * dxu_dx)
  tau = (tau / norm).T
  n = np.vstack((tau[:,1], -tau[:,0])).T
  xit = np.sum(-ap * tau, axis = 1)
  xin = np.sum(-ap * n, axis = 1)
  return xit, xin
   

################################################################################
def eval_local_coords_(p, X, n, tau):
  """ 
  Evaluate coordinates of p - Xu with respect to
  a system given by a tangential and normal vector to the curve
  :param p: array of length 2 (x and y coordinate) describing a point
  :param X: array of x,y coordinates of points
  :param n: array of normals at the points
  :param tau: array of tangential vectors at the points
  :return: coordinates of p in the local system given by (X, tau, n)
  """ 
  
  ap = X - p
  xit = np.sum(-ap * tau, axis = 1)
  xin = np.sum(-ap * n, axis = 1)
  return xit, xin
   

################################################################################
def deform_naca(mesh, bounds, params_ref, params,
                gen_dist_flag = False, plt_flag = False,
                wall_dist_flag = False,
                sampled_nodes_ix = None):
  """ 
  Deform the nodal coordinates inside a box
  :param mesh: list of mesh nodal coordinates, fluid topology and wall topology
  :param bounds: array of two x-coordinates and two y-coordinates of the 
                 box
  :param parms_ref: position of max camber, max camber, and thickness of the
                    reference airfoil
  :param parms: position of max camber, max camber, and thickness of the airfoil
  :param gen_dist_flag: if True, find the closest surfacepoints of the reference mesh 
                        and save them in dist.npy, otherwise load them from
                        dist.npy
  :param plt_flag: plot the airfoils
  """ 

  no, fluid, stick = mesh
  xb, yb = bounds
  P, M, T = params_ref
  P_2, M_2, T_2 = params

  if stick is not None:
     wall = np.unique(stick) - 1
  
#  x = np.concatenate((
#     np.linspace(0.0, 0.1, 10000)**2,
#     (np.linspace(0.01, 0.05, 20000))[1:],
#     (np.linspace(0.05, 0.90, 85000))[1:],
#     (np.linspace(0.90, 0.999, 80000))[1:],
#     (np.linspace(0.999, 1.0, 1000))[1:])
#     )
#  xarc = np.linspace(0.0, 1.0, 100000)
  x = np.concatenate((
     np.linspace(0.0, 0.1, 20000)**2,
     (np.linspace(0.01, 0.05, 100000))[1:],
     (np.linspace(0.05, 0.90, 500000))[1:],
     (np.linspace(0.90, 0.999, 100000))[1:],
     (np.linspace(0.999, 1.0, 4000))[1:])
     )
  xarc = np.linspace(0.0, 1.0, 400000)
  
  xu, yu, dxu_dx, dyu_dx = naca_point_and_tangent(x[:-1], P, M, T, 'upper')
  xl, yl, dxl_dx, dyl_dx = naca_point_and_tangent(x[1:-1], P, M, T, 'lower')
  xa, ya, dxa_dx, dya_dx, xc1, xc2, phi1, phi2 = \
     naca_point_and_tangent(xarc, P, M, T, 'arc')


  if plt_flag: 
    step = int(10)
    plt.figure(figsize=(8,5))
    plt.plot(np.concatenate((xl[::step], np.array([xl[-1]]))),
             np.concatenate((yl[::step], np.array([yl[-1]]))), 'b-+')
    plt.plot(np.concatenate((xu[::step], np.array([xu[-1]]))),
             np.concatenate((yu[::step], np.array([yu[-1]]))), 'r-+')
    plt.plot(xa[::step], ya[::step], 'm-+')
  
    p = []
    
    for i in range(len(p)):
      ap = np.vstack((xu, yu)).T - p[i]
      n = np.vstack((dyu_dx, -dxu_dx))
      norm = np.linalg.norm(n, axis = 0)
      n = (n / norm).T
      d = np.linalg.norm(ap - (n.T * np.sum(ap * n, axis = 1)).T, axis = 1)
      ix = np.argmin(d)
      print(ix, d[ix], np.linalg.norm(np.array([xu[ix],yu[ix]])-p[i]))
      plt.plot([p[i,0], xu[ix]], [p[i,1], yu[ix]], 'r--')
    
    for i in range(len(p)):
      ap = np.vstack((xl, yl)).T - p[i]
      n = np.vstack((dyl_dx, -dxl_dx))
      norm = np.linalg.norm(n, axis = 0)
      n = (n / norm).T
      d = np.linalg.norm(ap - (n.T * np.sum(ap * n, axis = 1)).T, axis = 1)
      ix = np.argmin(d)
      print(ix, d[ix], np.linalg.norm(np.array([xl[ix],yl[ix]])-p[i]))
      plt.plot([p[i,0],xl[ix]], [p[i,1], yl[ix]], 'b--')
    
    for i in range(len(p)):
      ap = np.vstack((xa, ya)).T - p[i]
      n = np.vstack((dya_dx, -dxa_dx))
      norm = np.linalg.norm(n, axis = 0)
      n = (n / norm).T
      d = np.linalg.norm(ap - (n.T * np.sum(ap * n, axis = 1)).T, axis = 1)
      ix = np.argmin(d)
      print(ix, d[ix], np.linalg.norm(np.array([xa[ix],ya[ix]])-p[i]))
      plt.plot([p[i,0],xa[ix]], [p[i,1], ya[ix]], 'm--')
    
    plt.plot(no[wall,0], no[wall,1], 'kx')
 
  in_bounds = (no[:,0] >= xb[0]) & (no[:,0] <= xb[1]) & \
              (no[:,1] >= yb[0])  & (no[:,1] <= yb[1])
  ixbs = np.nonzero(in_bounds)[0]

  xs = [xu, xl, xa]
  ys = [yu, yl, ya]
  dxsdx = [dxu_dx, dxl_dx, dxa_dx]
  dysdx = [dyu_dx, dyl_dx, dya_dx]

  if gen_dist_flag:
    
    def compute_dist(s, no, xs, ys, dxsdx, dysdx, flip_normal):
      #distl = np.zeros(len(no))
      #distp = np.zeros(len(no))
      #closest_ix = -np.ones(len(no), dtype = np.int32)
      #for i in range(len(no)):
      #  distl[i], distp[i], closest_ix[i] = \
      #     eval_dist(no[i,:2], xs, ys, dxsdx, dysdx, flip_normal)
      #  if i % 1000 == 0:
      #    print('Done ', i, flush = True)
      iis = [i for i in range(len(no))]
      distl, distp, closest_ix = zip(*Parallel(n_jobs = 48) \
            (delayed(eval_dist)(no[i,:2], xs, ys, dxsdx, dysdx, flip_normal)
            for i in iis))
      distl = np.array(distl)
      distp = np.array(distp)
      closest_ix = np.array(closest_ix, dtype = np.int32)
      return distl, distp, closest_ix
    
    js = [0,1,2]
    flip_normal = [True, False, False]
    #res = Parallel(n_jobs = 3) \
    #      (delayed(compute_dist)(j, no[ixbs,:2],
    #       xs[j], ys[j], dxsdx[j], dysdx[j], flip_normal[j]) for j in js)
    res = []
    for j in js:
      tres = compute_dist(j, no[ixbs,:2],
           xs[j], ys[j], dxsdx[j], dysdx[j], flip_normal[j])
      res.append(list(tres))
      print('Done surface ', j, flush = True)
    
    output_file = open('mesh/dist.npy','wb') 
    for i in range(3):
      np.save(output_file, res[i][0], allow_pickle = False)
      np.save(output_file, res[i][1], allow_pickle = False)
      np.save(output_file, res[i][2], allow_pickle = False)
    output_file.close()
  
  else:
 
    input_file = open('mesh/dist.npy','rb') 
    res = []
    for i in range(3):
      tmp = []
      tmp.append(np.load(input_file))
      tmp.append(np.load(input_file))
      tmp.append(np.load(input_file))
      res.append(tmp)
    input_file.close()

  xu_2, yu_2, dxu_dx_2, dyu_dx_2 = \
    naca_point_and_tangent(x[:-1], P_2, M_2, T_2, 'upper')
  xl_2, yl_2, dxl_dx_2, dyl_dx_2 = \
    naca_point_and_tangent(x[1:-1], P_2, M_2, T_2, 'lower')
  xa_2, ya_2, dxa_dx_2, dya_dx_2, dummy, dummy, dummy, dummy = \
    naca_point_and_tangent(xarc, P_2, M_2, T_2, 'arc')
  xs_2 = [xu_2, xl_2, xa_2]
  ys_2 = [yu_2, yl_2, ya_2]
  
  if plt_flag: 
    plt.plot(np.concatenate((xl_2[::step], np.array([xl_2[-1]]))),
             np.concatenate((yl_2[::step], np.array([yl_2[-1]]))), 'b:')
    plt.plot(np.concatenate((xu_2[::step], np.array([xu_2[-1]]))),
             np.concatenate((yu_2[::step], np.array([yu_2[-1]]))), 'r:')
    plt.plot(xa_2[::step], ya_2[::step], 'm:')
  
  newno = np.copy(no)

  if sampled_nodes_ix is not None:
    #ixbs = np.intersect1d(ixbs, sampled_nodes_ix, assume_unique = True)
    ixbs, inter_ixbs_ix, inter_samp_ix = np.intersect1d(ixbs, sampled_nodes_ix, assume_unique = True, return_indices=True)
    for j in range(3):
        for k in range(3):
            res[j][k]=res[j][k][inter_ixbs_ix]

  for i in range(len(ixbs)):
    distl = np.zeros(3)#2)
    distp = np.zeros(3)#2)
    for s in range(3):#2):
      distl[s] = res[s][0][i]
      distp[s] = res[s][1][i]
    ixs = np.argsort(distp) 
  
    # Switching to a geometric test for points in the arc area
    phi = np.arctan2(-(xc2-no[ixbs[i],1]),-(xc1-no[ixbs[i],0]))
    if False: #no[ixbs[i],0] > xc1 and phi <= phi1 and phi >= phi2:
      ix = 2
    else:
      if distl[ixs[0]] < 6.0e-6:
        ix = ixs[0]
      elif distl[ixs[1]] < 6.0e-6:
        ix = ixs[1]
        print('Second surface selected', no[ixbs[i],:2])
        print(distl)
        print(distp)
        if plt_flag: 
          plt.plot([no[ixbs[i],0]],[no[ixbs[i],1]],'c.')
      else:
        ix = ixs[0]
        print("No point is close enough: ",
                         distl, distp,no[ixbs[i],:], res[ix][2][i])
        #raise ValueError("No point is close enough: ",
        #                 distl, distp,no[ixbs[i],:])
   
    # Closest surface  point index 
    isp = res[ix][2][i]
    # Closest point on the surface and its normal
    p = np.array([ xs[ix][isp], ys[ix][isp] ])
    n = np.array([ dysdx[ix][isp], -dxsdx[ix][isp] ])
    if ix == 0 or ix == 2:
      n = -n
    n /= np.linalg.norm(n)
    # Find normal line intersection with the box
    if n[1] == 0.0:
      if p[0] < 0.5:
        pb = np.array([ xb[0], p[1] ])
      else:
        pb = np.array([ xb[1], p[1] ])
    elif n[0] == 0.0:
      if p[1] < 0.0:
        pb = np.array([ p[0], yb[0] ])
      else:
        pb = np.array([ p[0], yb[1] ])
    else:
      ts = np.array([(xb[0] - p[0]) / n[0], (xb[1] - p[0]) / n[0], 
                     (yb[0] - p[1]) / n[1], (yb[1] - p[1]) / n[1] ])
      it = np.argsort(ts)
      for iit in range(4):
        if ts[it[iit]] > 0.0:
          break 
      pb = p + ts[it[iit]] * n 
  
    # Compute radial coordinate
    xi = np.dot(no[ixbs[i],:2] - p,n) / np.linalg.norm(pb - p)
  
    # Surface point on the new surface 
    p_2 = np.array([ xs_2[ix][isp], ys_2[ix][isp] ])
    newno[ixbs[i],:2] = p_2 + xi * (pb - p_2) 

  if plt_flag: 
    plt.plot(newno[wall,0], newno[wall,1], 'go')
    plt.axis('equal')
    plt.show()

  if wall_dist_flag: 
    #dm = distance_matrix(newno[wall,:2], newno[sampled_nodes_ix,:2], 2, threshold = 100000000)
    if sampled_nodes_ix is not None:
        dm = distanceMatrixPower(newno[wall,:2], newno[sampled_nodes_ix,:2], 2)
    else:
        dm = distanceMatrixPower(newno[wall,:2], newno[:,:2], 2)
    wd = np.sqrt(np.min(dm, axis = 0))
  else:
    wd = []
  
  if sampled_nodes_ix is not None: 
    return newno[sampled_nodes_ix,:], wd
  else:
    return newno, wd

################################################################################
#ncpu = 240
#gen_dist_flag = True #True
#plt_flag = False
#run_decomp = False
#
##no, fluid, stick = load_mesh('no650', 'fluid650', 'stick650')
#no, fluid, stick = load_mesh('mesh/naca0012_Re1p5_nodes', 'mesh/naca0012_Re1p5_fluid', 'mesh/naca0012_Re1p5_stick')
##no, fluid, stick = load_mesh('no', None, None)
#
#wall_dist_flag = True #False
#sampled_nodes_ix = np.linspace(0, len(no)-1,num=40000,dtype=np.int32)
##sampled_nodes_ix = np.linspace(0, len(no)-1,num=len(no),dtype=np.int32)
#
#xb = np.array([-0.20, 1.50])
#yb = np.array([-0.50, 0.50])
#
#P = 0.4
#M = 0.000
#T = 0.12 #0.125
#
##Scan validity of meshes
#Ps = np.linspace(0.2,0.5,4)
#Ms = np.linspace(0.0,0.05,6)
#Ts = np.linspace(0.05,0.2,7)
#
#params = []
#for P_2 in Ps:
#  for M_2 in Ms:
#    for T_2 in Ts:
#       params.append([P_2, M_2, T_2])
#
#params = np.array([ [0.3, 0.05, 0.15] ])
#
##for irun in range(len(params)):
##  P_2, M_2, T_2 = params[irun] 
##  newno = deform_naca([no, fluid, stick], [xb, yb], [P, M, T], [P_2, M_2, T_2], 
##              gen_dist_flag = gen_dist_flag, plt_flag = plt_flag)
##  # Create new naca.top
##  np.savetxt('newno',
##    np.hstack((np.linspace(1,len(newno), num = len(newno))[:,np.newaxis], newno)),
##    fmt = ['%d','%.10e', '%.10e','%.10e'])
#
#P_2, M_2, T_2 = params[-1] 
#newno, wd = deform_naca([no, fluid, stick], [xb, yb], [P, M, T],[P_2, M_2, T_2],
#       gen_dist_flag = gen_dist_flag, plt_flag = plt_flag, wall_dist_flag = wall_dist_flag,
#       sampled_nodes_ix = sampled_nodes_ix)

