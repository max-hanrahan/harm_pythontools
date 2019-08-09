def Megantest(fnumber, ncell):
    """Function to compute and save the radial average of Reynolds and Maxwell Stresses and the vertical B field (lab frame) in the desired range from r1 to r2"""
    ###Megan's test function to be used in iPython  
    grid3d("gdump.bin",use2d=True) #load the gdump.bin file - use2d=True to save memory
    #make the spherical polar grid fully 3D
    myr3d=mk2d3d(r)
    myh3d=mk2d3d(h)
    myph3d=mk2d3d(ph)
    for k in range(0,nz):
	myph3d[:,:,k]=(0.5+k)*2*np.pi/nz
    #compute cartesian coordinates for each grid point of the array
    myx=myr3d*np.sin(myh3d)*np.cos(myph3d)
    myy=myr3d*np.sin(myh3d)*np.sin(myph3d)
    myz=myr3d*np.cos(myh3d)
    myxeq=myx[:,ny/2,:]
    myyeq=myy[:,ny/2,:]
    myzeq=myz[:,ny/2,:]
    #load the fieldline file for a given time and compute standard quantities
    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=np.floor(iofr(rhor)+0.5)
    #set parameters for the interpolation routine
    rng=40.0
    ncell=ncell
    extent=(-rng,rng,-rng,rng)
    dx=dy=dz=(rng-(-rng))/ncell #gives the grid size to use when finding magnetic flux
    #compute quantities used for masking (currently magnetic flux and inverse beta 2/24/16)
    pg=(gam-1.0)*ug
    myfun=0.5*bsq/pg #ibeta
    #flux_z=-gdetB[2]*_dx1*_dx3
    #flux_r=gdetB[1]*_dx2*_dx3
    #flux_zeq=np.sum(flux_z[:,ny/2-1:ny/2+1,:],axis=1)*0.5
    #flux_z_int=np.sum(flux_zeq[(myxeq>-rng) & (myxeq<rng) & (myyeq>-rng) & (myyeq<rng)]) #make sure cell center is within |x|,|y|<=40
    #flux_r_int=np.sum(flux_r[ihor,0:ny/2,:])
    mdot=gdet*uu[1]*_dx2*_dx3*rho
    mdot_int=np.sum(np.abs(mdot[ihor,:,:]))

    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #
    Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
    BRnorm=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
    #reinterpolate onto an evenly space grid on the equatorial plane
    imyx=reinterpxy(myx,extent,ncell,domask=0,interporder='linear')
    imyy=reinterpxy(myy,extent,ncell,domask=0,interporder='linear')
    imyz=reinterpxy(myz,extent,ncell,domask=1,interporder='linear')
    #replace z values that would be inside the black hole with those on the black hole horizon - need to not hard code this in case I change resolution
    imyz[imyz.mask==True]=np.sqrt(rhor**2-imyx[imyz.mask==True]**2-imyy[imyz.mask==True]**2)

    imyxhor=reinterpxyhor(myx,extent,ncell,domask=1,interporder='linear')
    imyyhor=reinterpxyhor(myy,extent,ncell,domask=1,interporder='linear')
    imyzhor=reinterpxyhor(myz,extent,ncell,domask=1,interporder='linear')

    imyfun=reinterpxy(myfun,extent,ncell,domask=1,interporder='linear')
    myfun_hor=reinterpxyhor(myfun,extent,ncell,domask=1,interporder='linear')
    imyfun[imyfun.mask==True]=myfun_hor[imyfun.mask==True] #replace values that are inside the black hole in the equatorial plane with those in the upper half of the horizon

    irho=reinterpxy(rho,extent,ncell,domask=1,interporder='linear')
    irho_h=reinterpxyhor(rho,extent,ncell,domask=1,interporder='linear')
    irho[irho.mask==True]=irho_h[irho.mask==True]

    iBz=reinterpxy(Bznorm,extent,ncell,domask=1,interporder='linear')
    iBr=reinterpxyhor(Bznorm,extent,ncell,domask=1, interporder='linear') #Brnorm is spherical, BRnorm is cylindrical; switching to Bznorm to keep transition to the disk continuous, has sign info
    R=np.sqrt(imyxhor**2+imyyhor**2)
    #gdetxy=2.0/np.sqrt(2*(1+np.sqrt(1-a**2))-a**2-R**2)
    #flux_R=gdetxy*iBr*dx*dy
    #flux_z=iBz*dx*dy
    #Brfluxtot=np.sum(gdetxy[iBr.mask==False]*iBr[iBr.mask==False]*dx*dy)
    #Bzfluxtot=np.sum(iBz[iBz.mask==False])*dx*dy
    #ratioflux=flux_r_int/Brfluxtot

    ahor=2.0*np.pi*(a**2+3*rhor**2)/3 #surface area of half of the horizon
    BzH=5.0/ahor #Magnetic field per unit area on BH horizon, calculated from Upsilon value in Avara 2015
    Bzfake=iBz*np.sqrt(imyx**2+imyy**2)/(rhor*np.sqrt(5.75)) #vertical magnetic field in the disk per unit area with radial dependence correction, Mdot=5.75 as in Avara 2015
    Br=iBr/np.sqrt(5.75) #radial magnetic field on the horizon per unit area, Mdot=5.75 as in Avara 2015
    Bzfake[iBz.mask==True]=Br[iBz.mask==True]#replace values that are inside the black hole in the equatorial plane with those in the upper half of the horizon
    Bzfake1=Bzfake/BzH

    #create mask using inverse beta cutoff and apply it to the other relevant quantities (position and flux)
    msk1=ma.masked_where(imyfun>10,imyfun)
    xmsk = imyx[msk1.mask==True]
    ymsk = imyy[msk1.mask==True]
    zmsk = imyz[msk1.mask==True]
    Bmsk = Bzfake1[msk1.mask==True]

    betap = imyfun[msk1.mask==True]
    betap1=np.copy(imyfun)
    #create an unnormalized probability function for inverse beta to better choose relevant seedpoints
    betap1=(1/99.0)*betap1-(1/99.0) #cutoffs set at 1,100 4/8/16
    betap1[betap1>1]=1
    betap1[betap1<0]=0
    
    #create an unnormalized probability function for B/sqrt(Mdot) to better choose relevant seedpoints; Still adjusting cutoffs 4/1/16
    Bprob=np.copy(Bmsk)
    Bprob=np.abs(Bprob)
    Bprob[Bprob<0.1]=0.1
    Bprob[Bprob>1.0]=1
    Bprob=1.11*Bprob - 0.11

    #Randomly select seedpoints by first finding coordinate points (x,y,z) then converting to grid spacing
    #get_coordsw(xmsk, ymsk, zmsk.data, Bprob, betap1, 20, prbmax, fnumber)
    #v5d_traj(fnumber,100,rng,nsp)

    #Make plots to use as movie frames
    plt.clf()
    plt.figure(1)
    ax=np.linspace(-rng, rng, ncell)
    plt.pcolor(ax,ax,betap1)
    plt.colorbar()
    plt.title('Inverse Beta Probability Distribution '+str(fnumber).zfill(4))
    plt.savefig('betaprob'+str(fnumber).zfill(4)+'new.png')
