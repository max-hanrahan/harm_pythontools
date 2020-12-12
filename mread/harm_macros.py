from matplotlib import rc

import numpy as np
import array
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import ma
import matplotlib.colors as colors
import os,glob
import pylab


def Qmri():
    """
    APPROXIMATELY Computes number of theta cells resolving one MIR wavelength
    """
    global bu,rho,uu,_dx2
    res=np.abs(bu[2]/np.sqrt(rho)/((uu[3]+1e-15)/uu[0])/_dx2)
    return(res)
def plco(myvar,xcoord=None,ycoord=None,**kwargs):
    plt.clf()
    plc(myvar,xcoord,ycoord,**kwargs)

def plc(myvar,xcoord=None,ycoord=None,**kwargs): #plc
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print(("The quantity you are trying to plot is a constant = %g." % np.min(myvar)))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    if( xcoord == None or ycoord == None ):
        res = plt.contour(myvar[:,:,0].transpose(),nc,**kwargs)
    else:
        res = plt.contour(xcoord[:,:,0],ycoord[:,:,0],myvar[:,:,0],nc,**kwargs)
    if( cb == True): #use color bar
        plt.colorbar()

def reinterp(vartointerp,extent,ncell):
    global xi,yi,zi
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    xraw=r*np.sin(h)
    yraw=r*np.cos(h)
    x=xraw[:,:,0].view().reshape(-1)
    y=yraw[:,:,0].view().reshape(-1)
    var=vartointerp[:,:,0].view().reshape(-1)
    #mirror
    x=np.concatenate((-x,x))
    y=np.concatenate((y,y))
    var=np.concatenate((vartointerp[:,:,0].view().reshape(-1),var))
    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    # grid the data.
    zi = griddata((x, y), var, (xi[None,:], yi[:,None]), method='linear')
    interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < 1+np.sqrt(1-a**2)
    #zi[interior] = np.ma.masked
    varinterpolated = ma.masked_where(interior, zi)
    return(varinterpolated)
    
def mkframe(fname,vmin=None,vmax=None):
    len = 60
    extent=(-len,len,-len,len)
    ncell=800
    palette=cm.jet
    palette.set_bad('k', 1.0)
    palette.set_over('r', 1.0)
    palette.set_under('g', 1.0)
    aphi = fieldcalc()
    iaphi = reinterp(aphi,extent,ncell)
    ilrho = reinterp(np.log10(rho),extent,ncell)
    #maxabsiaphi=np.max(np.abs(iaphi))
    maxabsiaphi = 100 #50
    ncont = 50 #30
    levs=np.linspace(-maxabsiaphi,maxabsiaphi,ncont)
    cset2 = plt.contour(iaphi,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower',levels=levs)
    #for c in cset2.collections:
    #    c.set_linestyle('solid')
    #CS = plt.contourf(xi,yi,zi,15,cmap=palette)
    CS = plt.imshow(ilrho, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower',vmin=vmin,vmax=vmax)
    #CS.cmap=cm.jet
    #CS.set_axis_bgcolor("#bdb76b")
    plt.colorbar(CS) # draw colorbar
    plt.xlim(extent[0],extent[1])
    plt.ylim(extent[2],extent[3])
    plt.title(r'$\log_{10}\rho$ at $t = %4.0f$' % t)
    plt.savefig( fname + '.png' )

def rrdump(dumpname):
    global nx,ny,nz,t,a,rho,ug,vu,vd,B,gd,numcols,gdetB
    #print( "Reading " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[0])
    ny = int(header[1])
    nz = int(header[2])
    t  = float(header[3])
    a  = float(header[6])
    gin.close()
    gd1 = np.loadtxt( "dumps/" + dumpname, 
                      dtype=float, 
                      skiprows=1, 
                      unpack = True )
    gd=gd1.view().reshape((-1,nx,ny,nz), order='F')
    rho,ug = gd[0:2,:,:,:].view() 
    B = np.zeros_like(gd[4:8])
    vu = np.zeros_like(B)
    vu[1:4] = gd[2:5].view() #relative 4-velocity only has non-zero spatial components
    B[1:4] = gd[5:8].view()
    numcols = gd.shape[0]  #total number of columns is made up of (n prim vars) + (n cons vars) = numcols
    gdetB = np.zeros_like(B)
    gdetB[1:4] = gd[numcols/2+5:numcols/2+5+3]  #gdetB starts with 5th conserved variable
    if 'gv3' in globals() and 'gn3' in globals(): 
        vd = mdot(gv3,vu)
        gamma = (1+mdot(mdot(gv3,vu),vu))**0.5
        etad = np.zeros_like(vu)
        etad[0] = -1/(-gn3[0,0])**0.5      #ZAMO frame velocity (definition)
        etau = mdot(gn3,etad)
        uu = gamma * etau + vu
        ud = mdot(gv3,uu)
    else:
        print( 'Metric (gv3, gn3) not defined, I am skipping the computation of uu and ud' )

   
def fieldcalcU():
    """
    Computes the field vector potential
    """
    daphi = mysum2(gdetB[1])*_dx2*_dx3
    aphi=daphi.cumsum(axis=1)
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi[0:nx-1] = 0.5*(aphi[0:nx-1]+aphi[1:nx]) #and in r
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalcface():
    """
    Computes the field vector potential
    """
    daphi = mysum2(gdetB[1])*_dx2*_dx3
    aphi=daphi.cumsum(axis=1)
    aphi-=daphi #correction for half-cell shift between face and center in theta
    #aphi[0:nx-1] = 0.5*(aphi[0:nx-1]+aphi[1:nx]) #and in r
    aphi[:,ny-1:ny/2:-1,:] = aphi[:,1:ny/2,:]
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalcface2():
    """
    Computes the field vector potential
    """
    daphi = np.sum(gdetB[2], axis=2)[:,:,None]*_dx1*_dx3
    aphi=daphi.cumsum(axis=0)
    aphi-=daphi #correction for half-cell shift between face and center in theta
    #aphi[0:nx-1] = 0.5*(aphi[0:nx-1]+aphi[1:nx]) #and in r
    aphi/=(nz*_dx3)
    return(aphi)

def rd(dump):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd
    global v1m,v1p,v2m,v2p,v3m,v3p,bsq
    #read image
    fin = open( "dumps/" + dump, "rb" )
    header = fin.readline().split()
    t = np.float64(header[0])
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gam=float(header[11])
    a=float(header[12])
    if dump.endswith(".bin"):
        body = np.fromfile(fin,dtype=np.double,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        fin.close()
    else:
        fin.close()
        gd = np.loadtxt( "dumps/"+dump, 
                      dtype=float, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:,:].view() 
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[11:14]
    B[1:4] = gd[14:17]
    pg,cs2,Sden = gd[17:20]
    U = gd[20:29]
    gdetB = np.zeros_like(B)
    gdetB[1:4] = U[5:8]
    divb = gd[29]
    uu = gd[30:34]
    ud = gd[34:38]
    bu = gd[38:42]
    bd = gd[42:46]
    bsq = mdot(bu,bd)
    v1m,v1p,v2m,v2p,v3m,v3p=gd[46:52]

def rgfd(fieldlinefilename,**kwargs):
    if not os.path.isfile(os.path.join("dumps/", fieldlinefilename)):
        print(( "File " + fieldlinefilename + " does not exist. Aborting." ))
        return
    if 'gv3' not in globals():
        gdumpname = glob.glob( os.path.join("dumps/", "gdump*") )
        #read the 1st found file
        grid3d(os.path.basename(gdumpname[0]))
    rfd(fieldlinefilename,**kwargs)
    cvel()
    

def rfd(fieldlinefilename,**kwargs):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,rho,lrho,ug,uu,uut,uu,B,uux,gdetB
    #read image
    fin = open( "dumps/" + fieldlinefilename, "rb" )
    header = fin.readline().split()
    t = np.float64(header[0])
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gam=float(header[11])
    a=float(header[12])
    body = np.fromfile(fin,dtype=np.single,count=-1)
    fin.close()
    d=body.view().reshape((-1,nx,ny,nz),order='F')
    #rho, u, -hu_t, -T^t_t/U0, u^t, v1,v2,v3,B1,B2,B3
    rho=d[0,:,:,:]
    lrho = np.log10(rho)
    ug=d[1,:,:,:]
    uu=d[4:8,:,:,:]  #note uu[i] are 3-velocities (as read from the fieldline file)
    #uut=np.copy(d[4,:,:,:])
    #multiply by u^t to get 4-velocities: u^i = u^t v^i
    #uux=np.copy(uu)
    #for i in range(1,4):
    #    uux[i,:,:,:] = uux[i,:,:,:] * uux[0,:,:,:]  
    uu[1:4]=uu[1:4] * uu[0]
    #old image format
    B = np.zeros_like(uu)
    B[1:4,:,:,:]=d[8:11,:,:,:]
    #if the input file contains additional data
    if(d.shape[0]>=14): 
        #new image format additionally contains gdet*B^i
        gdetB = np.zeros_like(B)
        gdetB[1:4] = d[11:14,:,:,:]
    else:
        print("No data on gdetB, approximating it.")
        gdetB = np.zeros_like(B)
        gdetB[1] = gdet * B[1]
        gdetB[2] = gdet * B[2]
        gdetB[3] = gdet * B[3]
    #     if 'gdet' in globals():
    #         #first set everything approximately (B's are at shifted locations by half-cell)
    #         B = gdetB/gdet  
    #         #then, average the inner cells to proper locations
    #         B[1,0:nx-1,:,:] = 0.5*(gdetB[1,0:nx-1,:,:]+gdetB[1,1:nx,:,:])/gdet[0:nx-1,:,:]
    #         B[2,:,0:ny-1,:] = 0.5*(gdetB[2,:,0:ny-1,:]+gdetB[2,:,1:ny,:])/gdet[:,0:ny-1,:]
    #         B[3,:,:,0:nz-1] = 0.5*(gdetB[3,:,:,0:nz-1]+gdetB[3,:,:,1:nz])/gdet[:,:,0:nz-1]
    #         #note: last cells on the grids (near upper boundaries of each dir are at
    #         #      approximate locations
    #     else:
    #         print( "rfd: warning: since gdet is not defined, I am skipping the computation of cell-centered fields, B" )
    # else:


def cvel():
    global ud,etad, etau, gamma, vu, vd, bu, bd, bsq
    ud = mdot(gv3,uu)                  #g_mn u^n
    etad = np.zeros_like(uu)
    etad[0] = -1/(-gn3[0,0])**0.5      #ZAMO frame velocity (definition)
    etau = mdot(gn3,etad)
    gamma=-mdot(uu,etad)                #Lorentz factor as measured by ZAMO
    vu = uu - gamma*etau               #u^m = v^m + gamma eta^m
    vd = mdot(gv3,vu)
    bu=np.empty_like(uu)              #allocate memory for bu
    #set component per component
    bu[0]=mdot(B[1:4], ud[1:4])             #B^i u_i
    bu[1:4]=(B[1:4] + bu[0]*uu[1:4])/uu[0]  #b^i = (B^i + b^t u^i)/u^t
    bd=mdot(gv3,bu)
    bsq=mdot(bu,bd)



def grid3d(dumpname): #read gdump: header and body
    global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
    print(( "Reading grid from " + "dumps/" + dumpname + " ..." ))
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    #read gdump
    #
    if dumpname.endswith(".bin"):
        body = np.fromfile(gin,dtype=np.double,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        gin.close()
    else:
        gin.close()
        gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=float, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    #gd = np.genfromtxt( "dumps/gdump", 
    #                 dtype=float, 
    #                 skip_header=1, 
    #                 skip_footer=nx*ny*(nz-1),
    #                 unpack = True ).view().reshape((137,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:,:].view() 
    #get the right order of indices by reversing the order of indices i,j(,k)
    #conn=gd[9:73].view().reshape((4,4,4,nx,ny,nz), order='F').transpose(2,1,0,3,4,5)
    gn3 = gd[73:89].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    gv3 = gd[89:105].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    gdet = gd[105]
    ck = gd[106:110].view().reshape((4,nx,ny,nz), order='F')
    dxdxp = gd[110:126].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    print( "Done!" )

def rdebug(debugfname):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a
    global fail0,floor0,limitgamma0,inflow0,failrho0,failu0,failrhou0,precgam0,precu0,toentropy0,tocold0,eosfail0
    global fail1,floor1,limitgamma1,inflow1,failrho1,failu1,failrhou1,precgam1,precu1,toentropy1,tocold1,eosfail1
    global fail2,floor2,limitgamma2,inflow2,failrho2,failu2,failrhou2,precgam2,precu2,toentropy2,tocold2,eosfail2
    global fail3,floor3,limitgamma3,inflow3,failrho3,failu3,failrhou3,precgam3,precu3,toentropy3,tocold3,eosfail3
    global dtot0, dtot1, dtot2, dtot3
    global lgdtot0, lgdtot1, lgdtot2, lgdtot3
    global failtot0, failtot1, failtot2, failtot3 
    global lgftot0, lgftot1, lgftot2, lgftot3 
    #read image
    fin = open( "dumps/" + debugfname, "rb" )
    header = fin.readline().split()
    t = np.float64(header[0])
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gam=float(header[11])
    a=float(header[12])
    if debugfname.endswith(".bin"):
        body = np.fromfile(fin,dtype=np.double,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        fin.close()
    else:
        fin.close()
        gd = np.loadtxt( "dumps/"+debugfname, 
                      dtype=float, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    (
       fail0,floor0,limitgamma0,inflow0,failrho0,failu0,failrhou0,precgam0,precu0,toentropy0,tocold0,eosfail0,
       fail1,floor1,limitgamma1,inflow1,failrho1,failu1,failrhou1,precgam1,precu1,toentropy1,tocold1,eosfail1,
       fail2,floor2,limitgamma2,inflow2,failrho2,failu2,failrhou2,precgam2,precu2,toentropy2,tocold2,eosfail2,
       fail3,floor3,limitgamma3,inflow3,failrho3,failu3,failrhou3,precgam3,precu3,toentropy3,tocold3,eosfail3
    ) = gd[0:48,:,:,:].view() 
 
    # shows where *ever* failed or not
    lg1fail=np.log10(fail0+1)
    lg1tot=np.log10(fail0+failrho0+failu0+failrhou0+1)
    #
    lg1precgam=np.log10(precgam0+1)
    lg1precu=np.log10(precu0+1)
    #
    failtot0=fail0+failrho0+failu0+failrhou0
    failtot1=fail1+failrho1+failu1+failrhou1
    failtot2=fail2+failrho2+failu2+failrhou2
    failtot3=fail3+failrho3+failu3+failrhou3
    #
    lgftot0=np.log10(failtot0+1)
    lgftot1=np.log10(failtot1+1)
    lgftot2=np.log10(failtot2+1)
    lgftot3=np.log10(failtot3+1)
    #
    failtot0sum=np.sum(failtot0)
    failtot1sum=np.sum(failtot1)
    failtot2sum=np.sum(failtot2)
    failtot3sum=np.sum(failtot3)
    #
    print(( "failtotsum(0,1,2,3): %10d, %10d, %10d, %10d" % (failtot0sum, failtot1sum, failtot2sum, failtot3sum) ))
    #
    # absolute totals
    dtot0=fail0+floor0+limitgamma0+failrho0+failu0+failrhou0+precgam0+precu0
    dtot1=fail1+floor1+limitgamma1+failrho1+failu1+failrhou1+precgam1+precu1
    dtot2=fail2+floor2+limitgamma2+failrho2+failu2+failrhou2+precgam2+precu2
    dtot3=fail3+floor3+limitgamma3+failrho3+failu3+failrhou3+precgam3+precu3
    #
    lgdtot0=np.log10(dtot0+1)
    lgdtot1=np.log10(dtot1+1)
    lgdtot2=np.log10(dtot2+1)
    lgdtot3=np.log10(dtot3+1)
    #
    dtot0sum=np.sum(dtot0)
    dtot1sum=np.sum(dtot1)
    dtot2sum=np.sum(dtot2)
    dtot3sum=np.sum(dtot3)
    #
    print(( "   dtotsum(0,1,2,3): %10d, %10d, %10d, %10d" % (dtot0sum, dtot1sum, dtot2sum, dtot3sum) ))
    #


def rfdgrid(dumpname): #read gdump: header and body
    global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
    print(( "Reading grid from " + "dumps/" + dumpname + " ..." ))
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gin.close()
    #read gdump
    #
    gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=float, 
                      skiprows=1, 
                      unpack = True,
                      usecols=(0,1,2,3,4,5,6,7,8)).view().reshape((-1,nx,ny,nz), order='F')
    #gd = np.genfromtxt( "dumps/gdump", 
    #                 dtype=float, 
    #                 skip_header=1, 
    #                 skip_footer=nx*ny*(nz-1),
    #                 unpack = True ).view().reshape((137,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph,gdet = gd[:,:,:,:].view() 

def mdot(a,b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
    """
    if a.ndim == 4 and b.ndim == 4:
          c = (a*b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
          c = np.empty_like(a[:,0,:,:,:])      
          for i in range(a.shape[0]):
                c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
          c = np.empty_like(b[0,:,:,:,:])      
          for i in range(b.shape[1]):
                c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    elif a.ndim == 5 and b.ndim == 5:
          c = np.empty((a.shape[0],b.shape[1],a.shape[2],a.shape[3],a.shape[4]))
          for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                      c[i,j,:,:,:] = (a[i,:,:,:,:]*b[:,j,:,:,:]).sum(0)
    else:
           raise Exception('mdot', 'wrong dimensions')
    return c

def fieldcalc():
    """
    Computes the field vector potential
    """
    #return((1-h[:,:,0]/np.pi)[:,:,None]*fieldcalcp()+(h[:,:,0]/np.pi)[:,:,None]*fieldcalcm())
    return(fieldcalcU())

def mysum2(vec):
    #return( vec[:,:,0][:,:,None]*nz )
    return( np.sum(vec, axis=2)[:,:,None] )

def horfluxcalc(ihor):
    """
    Computes the absolute flux through the sphere i = ihor
    """
    global gdetB, _dx2, _dx3
    #1D function of theta only:
    dfabs = (np.abs(gdetB[1,ihor])).sum(1)*_dx2*_dx3
    fabs = dfabs.sum(axis=0)
    #account for the wedge
    scaletofullwedge(fabs)
    #fabs *= 
    return(fabs)

def scaletofullwedge(val):
    return(val * 2*np.pi/(dxdxp[3,3,0,0,0]*nz*_dx3))

def mdotcalc(ihor):
    """
    Computes the absolute flux through the sphere i = ihor
    """
    #1D function of theta only:
    global gdet, rho, uu, _dx3, _dx3
    md = (-gdet[ihor]*rho[ihor]*uu[1,ihor]).sum()*_dx2*_dx3
    scaletofullwedge(md)
    return(md)


def diskfluxcalc(jmid,rmax=None):
    """
    Computes the absolute flux through the disk midplane at j = jmid
    """
    global gdetB,_dx1,_dx3,r
    #1D function of theta only:
    dfabs = (np.abs(gdetB[2,:,jmid,:])).sum(1)*_dx1*_dx3
    if rmax != None:
        dfabs = dfabs[r[:,0,0]<rmax]
    fabs = dfabs.sum(axis=0)
    scaletofullwedge(fabs)
    return(fabs)

def fhorvstime(ihor):
    """
    Returns a tuple (ts,fs,mdot): lists of times, horizon fluxes, and Mdot
    """
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    ts=np.empty(len(flist),dtype=float)
    fs=np.empty(len(flist),dtype=float)
    md=np.empty(len(flist),dtype=float)
    for findex, fname in enumerate(flist):
        print(( "Reading " + fname + " ..." ))
        rfd("../"+fname)
        fs[findex]=horfluxcalc(ihor)
        md[findex]=mdotcalc(ihor)
        ts[findex]=t
    print( "Done!" )
    return((ts,fs,md))

def plotit(ts,fs,md):
    #rc('font', family='serif')
    #plt.figure( figsize=(12,9) )
    fig,plotlist=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,9))
    #plt.subplots_adjust(hspace=0.4) #increase vertical spacing to avoid crowding
    plotlist[0].plot(ts,fs,label=r'$\Phi_{\rm h}/\Phi_{\rm i}$: Normalized Horizon Magnetic Flux')
    plotlist[0].plot(ts,fs,'r+') #, label=r'$\Phi_{\rm h}/0.5\Phi_{\rm i}$: Data Points')
    plotlist[0].legend(loc='lower right')
    #plt.xlabel(r'$t\;(GM/c^3)$')
    plotlist[0].set_ylabel(r'$\Phi_{\rm h}$',fontsize=16)
    plt.setp( plotlist[0].get_xticklabels(), visible=False)
    plotlist[0].grid(True)
    #
    #plotlist[1].subplot(212,sharex=True)
    plotlist[1].plot(ts,md,label=r'$\dot M_{\rm h}$: Horizon Accretion Rate')
    plotlist[1].plot(ts,md,'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
    plotlist[1].legend(loc='upper right')
    plotlist[1].set_xlabel(r'$t\;(GM/c^3)$')
    plotlist[1].set_ylabel(r'$\dot M_{\rm h}$',fontsize=16)
    
    #title("\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
    #      fontsize=16, color='r')
    plotlist[1].grid(True)
    fig.savefig('test.pdf')

def pl(x,y):
    plt.plot(x[:,ny/2,0],y[:,ny/2,0])

if __name__ == "__main__":
    import sys

    if False:
        #plot horizon magnetic flux and mdot vs. time from a series of fieldline dumps
        rgfd("fieldline0000.bin")
        diskflux=diskfluxcalc(ny/2)
        ts,fs,md=fhorvstime(11)
        plotit(ts,fs/(diskflux),md)

    if False:
        #check for failures example: read in debug0010 and plot failure locations; if using binary output, append ".bin" to the filename
        rdebug("debug0010")

        #plot critical failure locations (cumulative over all time)
        plt.figure()
        plco(fail0,cb=True)
        #plot critical failure locations (over last dump or image or sth else cycle)
        #replace fail1 with fail2 or fail3 to check different cycles
        plt.figure()
        plco(fail1,cb=True) #can replace fail1 with fail2 or fail3

        #plot all failure locations (see rdebug macro; cumulative over all time)        
        plt.figure()
        plco(dtot0,cb=True)

    if False:
        #make a movie of the simulation
        grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]) )
        flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
        for findex, fname in enumerate(flist):
            print(( "Reading " + fname + " ..." ))
            rfd("../"+fname)
            plt.clf()
            #this reinterpolates to a Cartesian grid and plots field lines
            #NOTE: may need to change hard-cored parameters like maxabsiaphi and ncont
            mkframe("lrho%04d" % findex, vmin=-8,vmax=0.2)
        print( "Done!" )
        print( "Now you can run to make the movie:");
        print( "ffmpeg -fflags +genpts -r 10 -i lrho%04d.png -vcodec mpeg4 -qmax 5 mov.avi")

