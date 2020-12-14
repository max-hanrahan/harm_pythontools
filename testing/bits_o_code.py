###Random bits of code that come in handy

###ASTRORAY stuff
I=np.load('figures/60fdiff/Ivstime_faceon_fdiff60.npy')

for i in range(11863,15008):
    fp=np.loadtxt('60fdiff/spectra/faceon/poliresa50th296fn'+str(i)+'case17017317boo_all.dat')
    I[0,i-1]=fp[0]
    I[1,i-1]=fp[2]

for i in range(16750,17258):
    fp=np.loadtxt('60fdiff/spectra/faceon/poliresa50th296fn'+str(i)+'hi_all.dat')
    I[0,i-1]=fp[0]
    I[1,i-1]=fp[2]

I[0,9895]=9896
I[0,11861]=11862
I[0,15007]=15008

np.save('figures/60fdiff/Ivstime_faceon_fdiff60.npy',I)

fig,ax=plt.subplots(1,4)
fig.set_size_inches(12.5,3.5)
#avg 3x5 stuff
se=ax[1,3].imshow(eavg_I13400[25:126,25:126,0], extent=[X[25],X[126],X[25],X[126]],vmax=eavg_I16750[25:126,25:126,0].max(),cmap=cmap.inferno)
ax[0,3].set_title(r"$t\approx 191 hours$")
ax[0,0].set_ylabel(r"$y\ [\mu as]$",fontsize=16,ha='center')
ax[2,0].set_xlabel(r"$x\ [\mu as]$",fontsize=16,ha='center')
cbar_ax0=fig.add_axes([0.92,0.68,0.01,0.2]) #good
cbar_ax1=fig.add_axes([0.92,0.4,0.01,0.2]) #good
cbar_ax2=fig.add_axes([0.92,0.125,0.01,0.2]) #good
plt.colorbar(se,cax=cbar_ax0,format='%.0e')  
#snap 1x4 stuff
ax[0].imshow(fsnap7754[25:126,25:126,0], extent=[X[25],X[126],X[25],X[126]],vmax=fsnap7754[25:126,25:126,0].max(),cmap=cmap.inferno)
ax[0].set_title(r"$t\approx 181 hours$")
ax[0].set_xlabel(r"$x\ [\mu as]$",fontsize=16,ha='center')
ax[1].imshow(fsnap7791[25:126,25:126,0], extent=[X[25],X[126],X[25],X[126]],vmax=fsnap7754[25:126,25:126,0].max(),cmap=cmap.inferno)
ax[1].set_title(r"$t\approx 182 hours$")
ax[1].set_xlabel(r"$x\ [\mu as]$",fontsize=16,ha='center')
ax[2].imshow(fsnap7936[25:126,25:126,0], extent=[X[25],X[126],X[25],X[126]],vmax=fsnap7754[25:126,25:126,0].max(),cmap=cmap.inferno)
ax[2].set_title(r"$t\approx 187 hours$")
ax[2].set_xlabel(r"$x\ [\mu as]$",fontsize=16,ha='center')
sf=ax[3].imshow(fsnap8200[25:126,25:126,0], extent=[X[25],X[126],X[25],X[126]],vmax=fsnap7754[25:126,25:126,0].max(),cmap=cmap.inferno)
ax[3].set_title(r"$t\approx 191 hours$")
ax[3].set_xlabel(r"$x\ [\mu as]$",fontsize=16,ha='center')
ax[0].set_ylabel(r"$y\ [\mu as]$",fontsize=16,ha='center')
cbar_ax0=fig.add_axes([0.92,0.15,0.01,0.7]) #good
plt.colorbar(sf,cax=cbar_ax0,format='%.0e')  
plt.savefig('fsnap_I_talk.pdf')

for ax in ax.flat:
    ax.label_outer()

###Getting random fieldlines from python to vis5d easily
#have coordinates chosen by get_coords(ibeta, ib_floor, m, fnumber) in __init__.py then run this for easy use

#fill=0
#s is from 0-7, gives set number/color
#m=number of seedpoints wanted
#lista is the list withe the chosen coordinates

for i in range(0,m):
    x,y,z=lista[i,0], 50.5, lista[i,1]
    x,y,z=101-lista[i,1], 50.5, lista[i,0]+1 #thinmad info
    row, col, lev = (ncell)-lista[i,1], lista[i,0], 50.5 #new scheme as of 12/1 - ncell is number of boxes across grid; v5d -> python switches x,y
    print 'vis5d_make_traj $dtx '+str(x)+' ' +str(y)+' ' +str(z)+' ' +str(fill) +' '+str(s)

###making a movie the dumb way with ffmpeg because it doesn't like -start_number
ffmpeg -r 1 -i RTmovie218%d.png -vcodec mjpeg movie1.mp4
#what about when I want to start with 2153? Renumber everything?

####find nearest values from array with x,y coords and fractional flux
test=array[:,2] #array of the flux values
#value is the number to match
def find_nearest(test,value):
    idx = (np.abs(test-value)).argmin()
    return idx

####get coordinates from choosing points on (-10,10) on a (-15,15) scale
no - change range in for loop

###combine different masks/filters
msk1=ma.masked_where(myfun>100,myfun)
msk2=ma.masked_where(bsor>5,myfun)
msk3=ma.masked_where(blob<0.33,myfun)
msk4=ma.masked_where(rho>5,myfun)
msktot=reduce(np.logical_and,(msk1,msk2,msk3,msk4))

#old version just using resolution grid
def get_coords_res(myfun,m,fnumber):
    '''Chooses a random sampling from a list of all coordinates (in Cartesian coordinates) that meet a certain qualification'''
    """Megan added 9/17/15"""
    coords_tot=np.array([[0,0]])
    coords_sampled=np.array([[0,0]])
    
    msktot=ma.masked_where(myfun>100,myfun)
    for x in range(0,100):
        for y in range(0,100):                                                        
            if msktot.mask[x,y]==True:
                a=np.array([[x,y]])
                coords_tot=np.concatenate((coords_tot,a),axis=0)

    coords_tot=np.delete(coords_tot,(0),axis=0)
    a=len(coords_tot)

    samp=np.random.choice(a, m, replace=False)
    for i in range(0,m):
        c=np.array([[coords_tot[i]]])
        coords_sampled=np.concatenate((coords_sampled,c),axis=0)

    coords_sampled=np.delete(coords_sampled,(0),axis=0)

    coords=open("coords"+str(fnumber)+"test.npz","w")
    np.savez(coords,coords_tot=coords_tot, coords_sampled=coords_sampled,samp=samp)
    coords.close()

#old weighted coordinate chooser
def get_coordsfw(x,y,z,flux,m,fnumber):
    '''Chooses a random sampling from a list of all coordinates (in Cartesian coordinates) that meet a certain qualification, weighted by magnetic flux'''
    """Megan added 9/17/15; updated 12/3/15"""
    coords_sampled=np.array([[0,0,0]])
    ftot=np.sum(flux)
    flux_list=flux/ftot

    a=len(flux_list)

    for i in range(1,a):
        flux_list[i]=flux_list[i-1]+flux_list[i]

    samp=np.random.random_sample(m)
    for i in range(0,m):
        for j in range(1,a):
            if flux_list[j-1]<=samp[i]<=flux_list[j]:
                c=np.array([[x[j],y[j],z[j]]])
                coords_sampled=np.concatenate((coords_sampled,c),axis=0)

    coords_sampled=np.delete(coords_sampled,(0),axis=0)

    coords=open("coords"+str(fnumber)+"fw_test.npz","w")
    np.savez(coords,x=x.data, y=y.data, z=z.data, coords_sampled=coords_sampled,flux_list=flux_list.data,samp=samp)
    coords.close()

#Old simple version
def get_coords(x,y,z,m,fnumber):
    '''Chooses a random sampling from a list of all coordinates (in Cartesian coordinates) that meet a certain qualification'''
    """Megan added 9/17/15"""
    '''x,y,z are masked arrays with Cartesian coordinates
    m is the number of points you want to select
    fnumber is the fieldline file number being used'''
    coords_sampled=np.array([[0,0,0]])
    a=len(x)

    samp=np.random.choice(a, m, replace=False)
    for i in range(0,m):
        c=np.array([[x[samp[i]],y[samp[i]],z[samp[i]]]])
        coords_sampled=np.concatenate((coords_sampled,c),axis=0)

    coords_sampled=np.delete(coords_sampled,(0),axis=0)

    coords=open("coords"+str(fnumber)+"test.npz","w")
    np.savez(coords,x=x.data, y=y.data, z=z.data, coords_sampled=coords_sampled,samp=samp)
    coords.close()

#work in progress
def Bcheck():
    #read in B
    grid3d("gdump.bin",use2d=True)
    myr3d=mk2d3d(r)
    myh3d=mk2d3d(h)
    myph3d=mk2d3d(ph)
    for k in range(0,nz):
        myph3d[:,:,k]=(0.5+k)*2*np.pi/nz

    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    # convert from x^(i) to B^{r,h,ph}
    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #convert from B^{r,h,ph} to B^{x,y,z}
    Bx = np.sin(myh3d)*np.cos(myph3d)*Br + np.cos(myh3d)*np.cos(myph3d)*Bh - np.sin(myph3d)*Bp
    By = np.sin(myh3d)*np.sin(myph3d)*Br + np.cos(myh3d)*np.sin(myph3d)*Bh + np.cos(myph3d)*Bp
    Bz = np.cos(myh3d)*Br - np.sin(myh3d)*Bh
    
    return Bx, By, Bz

def checkspdrop(fnumber):
    fc=np.load("coords"+str(fnumber)+"weighted.npz")
    pb=fc['prob']
    print len(pb)
    if len(pb)<30:
        #break                                                                  
        print fnumber

def fixprobarray(fnumber,ncell):
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
    #load the fieldline file for a given time and compute standard quantities                                                                                                                                        
    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=np.floor(iofr(rhor)+0.5)
    #set parameters for the interpolation routine                                                                                                                                                                    
    rng=40.0
    ncell=ncell
    extent=(-rng,rng,-rng,rng)
    #compute quantities used for masking (currently magnetic flux and inverse beta 2/24/16)                                                                                                                          
    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #                                                                                                                                                                                                                
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #                                                                                                                                                                                                                
    Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
#reinterpolate onto an evenly space grid on the equatorial plane                                                                                                                                                 
    imyx=reinterpxy(myx,extent,ncell,domask=0,interporder='linear')
    imyy=reinterpxy(myy,extent,ncell,domask=0,interporder='linear')

    iBz=reinterpxy(Bznorm,extent,ncell,domask=1,interporder='linear')
    iBr=reinterpxyhor(Bznorm,extent,ncell,domask=1, interporder='linear') #Brnorm is spherical, BRnorm is cylindrical; switching to Bznorm to keep transition to the disk continuous, has sign info                                                                                                                                                                                                                
    ahor=2.0*np.pi*(a**2+3*rhor**2)/3 #surface area of half of the horizon                                                                                                                                           
    BzH=5.0/ahor #Magnetic field per unit area on BH horizon, calculated from Upsilon value in Avara 2015                                                                                                            
    Bzfake=iBz*np.sqrt(imyx**2+imyy**2)/(rhor*np.sqrt(5.75)) #vertical magnetic field in the disk per unit area with radial dependence correction, Mdot=5.75 as in Avara 2015                                                                                                                                                                                                         
    Br=iBr/np.sqrt(5.75) #radial magnetic field on the horizon per unit area, Mdot=5.75 as in Avara 2015                                                                                                             
    Bzfake[iBz.mask==True]=Br[iBz.mask==True]#replace values that are inside the black hole in the equatorial plane with those in the upper half of the horizon                                                                                                                                                                                                             
    Bzfake1=Bzfake/BzH
    Bzfake1=np.abs(Bzfake1)
    #create an unnormalized probability function for B/sqrt(Mdot) to better choose relevant seedpoints; Bprob is 1D array from get_coordsw() while Bp_slice is 2D for python frames                                  
    Bp_slice=np.copy(Bzfake1)
    Bp_slice=(10*Bp_slice - 1)/9.0 #linear probability distribution between Bp=10 and Bp=100                                                                                                                         
    Bp_slice[Bp_slice>1.0]=1.0
    Bp_slice[Bp_slice<0.0]=0.0

    cf=np.load("june8/coords"+str(fnumber)+"weighted.npz")
    cs=cf['cs']
    lnc=int(len(cs[:,0]))
    prob_array=np.zeros(lnc)
    for n in range(0,lnc):
        print n
        x=cs[n,0]
        y=cs[n,1]
        grid=np.linspace(-rng,rng,300)
        iofx=getnearpos(grid,x)
        jofy=getnearpos(grid,y)
        prob_array[n]=Bp_slice[jofy,iofx]
        print "the probablity is %.2f" % prob_array[n]
    coords=open("june8/coords"+str(fnumber)+"w_100.npz","w")
    np.savez(coords,cs=cs, prob=prob_array)
    coords.close()
    #return Bp_slice,prob_array 

#pseudocode for trajectory tracking
do stuff in Megantest(fnum, res) up through Bzfake1
make Bprob,x,y arrays for full slice
ptot=np.sum(Bprob)
nsp=int(max(round(m*min(ptot/prbmax,1),0),10))

previousseed=checkifseedpointexists()

if previousseed==1:
    oldcoords=np.load('coords'+str(fnum-1)+'weighted.npz')
    xold=oldcoords['x']
    yold=oldcoords['y']
    vx = stuff from Bxnorm in dostreamlines if streamlines =/=1 #need to figure this out
    vy = stuff from Bynorm in dostreamlines if streamlines =/=1
    xnew = xold+vx*\delta_t
    ynew = yold+vy*\delta_t
    prob_array=np.zeros(int(len(xnew)))
    for n in range(0,int(len(xnew))):
        find nearest neighbor for xnew[n], ynew[n] #figure out how to do this
        iofx=arg(xnew) #figure out how to do this
        jofy=arg(ynew)
        prob_array[n] = Bprob[iofx,jofy,:]
        if prob_array[n]<=0:
            get new random seedpoint #modify get_coordsw to do variable number of seedpoints
            xnew[n]=xofnewpoint
            ynew[n]=yofnewpoint
            prob_array[n]=Bprob[xnew,ynew,0]
    sort prob_array from largest to smallest #figure out how to do this
    if len(xnew)>nsp:
        xnew=xnew[0:nsp]
        ynew=ynew[0:nsp]
    elif len(xnew)<nsp:
        get nsp-len(xnew) random seedpoints
        concatenate these to the bottom of xnew, ynew
        save xnew, ynew to file
        get v5d_traj()
else:
    get_coordsw(nsp)

vr = dxdxp[1,1]*uradu[1]/uradu[0]+dxdxp[1,2]*uradu[2]/uradu[0]
vh = dxdxp[2,1]*uradu[1]/uradu[0]+dxdxp[2,2]*uradu[2]/uradu[0]
vp = uradu[3]/uradu[0]*dxdxp[3,3]
#
#
vrnorm=vr
vhnorm=vh*np.abs(r)
vpnorm=vp*np.abs(r*np.sin(h))
#
vznorm=vrnorm*np.cos(h)-vhnorm*np.sin(h)
vRnorm=vrnorm*np.sin(h)+vhnorm*np.cos(h)
vxnorm=vRnorm*np.cos(ph)-vpnorm*np.sin(ph)
vynorm=vRnorm*np.sin(ph)+vpnorm*np.cos(ph)

def plot_stress(fnumber, phi, cap, floor, xy=True, noavg=True):
    '''Plotting function to visualize 2d slices of the disc                                                                                                                                                  
    fnumber = which fieldline file to load                                                                                                                                                                   
    phi = index of position to take slice at                                                                                                                                                                 
    cap = upper bound on quantity being plotted (if unwanted, use absurdly high value)                                                                                                                       
    floor = lower bound on quantity being plotted (if unwanted, use absurdly low value)                                                                                                                      
    IC = True: using the internal coordinates (Boyer-Lindquist)                                                                                                                                              
    IC = False: use spherical polar coordinates                                                                                                                                                              
    stress = True: plots stress terms (currently just Maxwell 6/12/15)                                                                                                                                       
    stress = False: plots vertical magnetic field                                                                                                                                                            
    radii of slice chosen with nxin, nxout'''

    # first load grid file
    global use2dglobal
    use2dglobal=True
    grid3d("gdump.bin", use2d=use2dglobal)
    # now try loading a single fieldline file                                                                        
    rfd("fieldline"+str(fnumber)+".bin")
    # now plot something you read-in                                                                                 
    plt.clf()
    plt.figure(1)
    #                                                                                                                
    ###############################                                                                                  
    if 1==1:
        (rhoclean,ugclean,uublob,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)=getrhouclean(rho,ug,uu)
        cvel()                                                     
        #                                                                                                            
        diskcondition=condmaxbsqorho
        # only around equator, not far away from equator                                                             
        diskcondition=diskcondition*(bsq/rho<1.0)*(np.fabs(h-np.pi*0.5)<0.1)                                                                 
        diskeqcondition=diskcondition                                                                                                                                                                       
    #
    ##############################
    ###choose the radial extent of the plot
    nxin=iofr(5)
    nxout=iofr(30)
    if noavg:
        numMag=jabs(-bu[1]*np.sqrt(gv3[1,1])*bd[3]*np.sqrt(gn3[3,3]))
        numRey=jabs(rho*(uu[1])*np.sqrt(gv3[1,1])*(ud[3])*np.sqrt(gn3[3,3]))

    else:
        avgexists=checkiffullavgexists()
        if avgexists==1:
            loadavg()
            loadedavg=1
            numMag=jabs(-(bu[1]-avg_bu[1])*np.sqrt(gv3[1,1])*(bd[3]-avg_bd[3])*np.sqrt(gn3[3,3]))
            numRey=jabs(rho*(uu[1]-avg_uu[1])*np.sqrt(gv3[1,1])*(ud[3]-avg_ud[3])*np.sqrt(gn3[3,3]))
        else:
            print 'avg2d.npy was not found so no turbulent stress can be computed'
            break

        denMR=(bsq*0.5+(gam-1.0)*ug)
        arey=numRey/denMR
        amag=numMag/denMR

        myfun=r*amag#+arey                                                                                                                                                                      
        myfun[myfun<=floor]=floor                                                                                   
        myfun[myfun>=cap]=cap
    ###########################
    ####choose coordinate system to plot
    if xy:
        ###using slice through equatorial plane
        myx=r[nxin:nxout,ny/2,:]*np.sin(h[nxin:nxout,ny/2,:])*np.cos(ph[nxin:nxout,ny/2,:])
        myy=r[nxin:nxout,ny/2,:]*np.sin(h[nxin:nxout,ny/2,:])*np.sin(ph[nxin:nxout,ny/2,:])
        myz=r[nxin:nxout,ny/2,:]*np.cos(h[nxin:nxout,ny/2,:])

        ax = plt.gca()
        ax.pcolor(myx,myy,myfun[nxin:nxout,ny/2,:],norm=[None,MidpointNormalize(midpoint=0)][ax>=1])  #try pcolormesh -faster; what does the norm[] part do?
        plc(myfun[nxin:nxout,ny/2,:],xcoord=myx,ycoord=myy,ax=ax,cb=True,nc=50,norm=[None,MidpointNormalize(midpoint=0)][ax>=1]) #nc = number of contour
        plt.savefig('amag_f'+str(fnumber)+'xy+'.png')
    else:
        ###use slice in phi
        myx=r[nxin:nxout,:,0]*np.sin(h[nxin:nxout,:,0])*np.cos(ph[nxin:nxout,:,0])
        myy=r[nxin:nxout,:,0]*np.sin(h[nxin:nxout,:,0])*np.sin(ph[nxin:nxout,:,0])
        myz=r[nxin:nxout,:,0]*np.cos(h[nxin:nxout,:,0])
        ax = plt.gca()
        ax.pcolor(myx,myz,myfun[nxin:nxout,:,phi])  #try pcolormesh - faster
        plc(myfun[nxin:nxout,:,phi],xcoord=myx,ycoord=myz,ax=ax,cb=True,nc=50) #nc = number of contour
        plt.savefig('amag_f'+str(fnumber)+'xz+'.png')
    #############################


def handpicking(fnumber,ncell):
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
    #load the fieldline file for a given time and compute standard quantities
    rfd("fieldline"+str(fnumber)+".bin")
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=np.floor(iofr(rhor)+0.5)
    #set parameters for the interpolation routine
    rng=40.0
    ncell=ncell
    extent=(-rng,rng,-rng,rng)
    #compute quantities used for masking (currently magnetic flux and inverse beta 2/24/16)
    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #
    Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
    #reinterpolate onto an evenly space grid on the equatorial plane
    imyx=reinterpxy(myx,extent,ncell,domask=0,interporder='linear')
    imyy=reinterpxy(myy,extent,ncell,domask=0,interporder='linear')

    iBz=reinterpxy(Bznorm,extent,ncell,domask=1,interporder='linear')
    #iBr=reinterpxyhor(Bznorm,extent,ncell,domask=1, interporder='linear') #Brnorm is spherical, BRnorm is cylindrical; switching to Bznorm to keep transition to the disk continuous, has sign info
    ahor=2.0*np.pi*(a**2+3*rhor**2)/3 #surface area of half of the horizon
    BzH=5.0/ahor #Magnetic field per unit area on BH horizon, calculated from Upsilon value in Avara 2015
    Bzfake=iBz*np.sqrt(imyx**2+imyy**2)/(rhor*np.sqrt(5.75)) #vertical magnetic field in the disk per unit area with radial dependence correction, Mdot=5.75 as in Avara 2015
    #Br=iBr/np.sqrt(5.75) #radial magnetic field on the horizon per unit area, Mdot=5.75 as in Avara 2015
    #Bzfake[iBz.mask==True]=Br[iBz.mask==True]#replace values that are inside the black hole in the equatorial plane with those in the upper half of the horizon
    Bzfake1=Bzfake/BzH
    Bzfake1=np.abs(Bzfake1)
    #create an unnormalized probability function for B/sqrt(Mdot) to better choose relevant seedpoints; Bprob is 1D array from get_coordsw() while Bp_slice is 2D for python frames
    Bp_slice=np.copy(Bzfake1)
    bmin=0.1 #value of Bz that I want to set to 0 in probability distribution
    bmax=np.max(Bp_slice) #value where the probability distribution goes to 1; value before 7/16 was 1.0
    sl=1/(bmax-bmin)
    yint=-bmin*sl
    Bp_slice=sl*Bp_slice+yint #linear probability distribution between Bp=10 and Bp=100
    Bp_slice[Bp_slice>1.0]=1.0
    Bp_slice[Bp_slice<0.0]=0.0
    Bprob=Bp_slice[msk1.mask==True]
    ptot=np.sum(Bprob)
    grid=np.linspace(-rng, rng, ncell)

    cf=np.load("june8/coords"+str(fnumber)+"weighted.npz")
    cs=cf['cs']
    hc=cf['hc']
    cf.close()

    #Make plots to use as movie frames
    plt.clf()
    plt.figure(1)
    plt.pcolor(grid,grid,Bp_slice)
    plt.colorbar()                                                        
    plt.scatter(cs[:,0],cs[:,1],color='fuchsia')
    plt.scatter(hc[:,0],hc[:,1],color='yellow')
    plt.xlim(-rng,rng)
    plt.ylim(-rng,rng)
    plt.title('Seedpoint propagation '+str(fnumber).zfill(4))
    #plt.show()
    arglist=[grid,grid,Bp_slice]
    argnamelist=["x","y","probability"]
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event,arglist,argnamelist,domask=domask))

def onclick(event,arglist,argnamelist,domask=None):
    if(domask==None):
        domask=1.0
    #thisline = event.artist
    #xdata2 = thisline.get_xdata()
    #ydata2 = thisline.get_ydata()
    xdata = event.xdata
    ydata = event.ydata
    #ind = event.ind
  # ind=%f zip=%f'%(
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(
        event.button, event.x, event.y, event.xdata, event.ydata);
    #global poo
    myx=event.xdata
    myz=event.ydata
    #
    extent=(myx*0.99,myx*1.01,myz*0.99,myz*1.01)
    ncell=2
    lenarglist=len(arglist)
    print('len(arglist)=%d' % (lenarglist))
    print(extent)
    print(Rin)
    
    for funi in range(lenarglist):
        funorig=arglist[funi]
        ifun = reinterp(funorig,extent,ncell,domask=domask,interporder='linear')
        print '%s[arg%d]=%f' %(argnamelist[funi],funi,ifun[0,0]);sys.stdout.flush()
#, event.ind, zip(xdata[ind],ydata[ind]))

def avgstr_v_time(fnumber):
    # first load grid file                                                                              
    global use2dglobal
    use2dglobal=True
    grid3d("gdump.bin", use2d=use2dglobal)
    # now try loading a single fieldline file
    rfd("fieldline"+str(fnumber).zfill(4)+".bin")
    ###############################                                                                                                                                                                                  
    (rhoclean,ugclean,uublob,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)=getrhouclean(rho,ug,uu)
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=iofr(rhor)
    ##############################                                                                                                                                                                                   
    ###choose the radial extent of the plot
    nxin=iofr(10)
    nxout=iofr(40)
    ###choose extent in r,theta:
    hmin=np.pi/2 - 0.1
    hmax=np.pi/2 + 0.1
    mhin=jofh(hmin,nxout)
    mhout=jofh(hmax,nxout)
    coremin=np.pi/2
    coremax=np.pi/2
    mc_in=jofh(coremin,nxout)
    mc_out=jofh(coremax,nxout)

    loadavg()
    br=bu[1]*np.sqrt(gv3[1,1])
    bphi=bd[3]*np.sqrt(gn3[3,3])
    nummag=-br*bphi
    ptot=0.5*avg_bsq+(gam-1.0)*avg_ug

    numavg=np.average(nummag*rho, axis=(1,2))
    denavg=np.average(ptot*rho, axis=(1,2))

    alpha=numavg/denavg
    return alpha

def stressvtime(fnumber):
    # first load grid file
    global use2dglobal
    use2dglobal=True
    grid3d("gdump.bin", use2d=use2dglobal)
    # now try loading a single fieldline file
    rfd("fieldline"+str(fnumber).zfill(4)+".bin")
    ###############################
    (rhoclean,ugclean,uublob,maxbsqorhonear,maxbsqorhofar,condmaxbsqorho,condmaxbsqorhorhs,rinterp)=getrhouclean(rho,ug,uu)
    cvel()
    rhor=1+(1-a**2)**0.5
    ihor=iofr(rhor)
    pg=(gam-1.0)*ug
    ##############################
    ###choose the radial extent of the plot
    nxin=iofr(10)
    nxout=iofr(40)
    ###choose extent in r,theta:
    hmin=np.pi/2 - 0.1
    hmax=np.pi/2 + 0.1
    mhin=jofh(hmin,nxout)
    mhout=jofh(hmax,nxout)
    coremin=np.pi/2
    coremax=np.pi/2
    mc_in=jofh(coremin,nxout)
    mc_out=jofh(coremax,nxout)

    loadavg()

    ###computing b-field components
    br=bu[1]*np.sqrt(gv3[1,1])
    brmean=avg_bu[1]*np.sqrt(gv3[1,1])
    brpert=br-brmean
    bz=-bu[2]*np.sqrt(gv3[2,2])
    bzmean=-avg_bu[2]*np.sqrt(gv3[2,2])
    bzpert=bz-bzmean
    bphi=bd[3]*np.sqrt(gn3[3,3])
    bphimean=avg_bd[3]*np.sqrt(gn3[3,3])
    bphipert=bphi-bphimean

    ###stress terms
    numtot=-br*bphi #full time dependent stress: should equal sum of other terms
    numtotvr=intangle(gdet[nxin:nxout,mhin:mhout,:]*numtot[nxin:nxout,mhin:mhout,:])
    numtotvt=np.sum(numtotvr*_dx1)

    num_mean=-brmean*bphimean #time averaged stress
    num_mean_vr=intangle(gdet[nxin:nxout,mhin:mhout,:]*num_mean[nxin:nxout,mhin:mhout,:])
    num_mean_vt=np.sum(num_mean_vr*_dx1)

    num_turb=-brpert*bphipert #turbulent compnent stress
    num_turb_vr=intangle(gdet[nxin:nxout,mhin:mhout,:]*num_turb[nxin:nxout,mhin:mhout,:])
    num_turb_vt=np.sum(num_turb_vr*_dx1)

    num_mix1=-brmean*bphipert #<br>*turbulent bphi
    num_mix1_vr=intangle(gdet[nxin:nxout,mhin:mhout,:]*num_mix1[nxin:nxout,mhin:mhout,:])
    num_mix1_vt=np.sum(num_mix1_vr*_dx1)

    num_mix2=-brpert*bphimean #turbulent br*<bphi>
    num_mix2_vr=intangle(gdet[nxin:nxout,mhin:mhout,:]*num_mix2[nxin:nxout,mhin:mhout,:])
    num_mix2_vt=np.sum(num_mix2_vr*_dx1)

    ###denominator
    ptot=0.5*avg_bsq+(gam-1.0)*avg_ug
    denom1=intangle(gdet[nxin:nxout,mhin:mhout,:]*ptot[nxin:nxout,mhin:mhout,:])
    denom=np.sum(denom1*_dx1)

    #normalizing
    alphatot=numtotvt/denom
    alphamean=num_mean_vt/denom
    alphaturb=num_turb_vt/denom
    alphamix1=num_mix1_vt/denom
    alphamix2=num_mix2_vt/denom

    '''myfun=(-br*bphi)*r**2/np.average(ptot[ihor,ny/2,:])
    myfun[myfun>3]=3
    myfun[myfun<-3]=-3
    ###########################   
    ###using slice through equatorial plane
    xy_x=r[nxin:nxout,ny/2,:]*np.sin(h[nxin:nxout,ny/2,:])*np.cos(ph[nxin:nxout,ny/2,:])
    myy=r[nxin:nxout,ny/2,:]*np.sin(h[nxin:nxout,ny/2,:])*np.sin(ph[nxin:nxout,ny/2,:])
    ###plot of myfun in xy plane
    plt.figure(1)
    plt.clf()
    ax = plt.gca()
    ax.pcolor(xy_x,myy,myfun[nxin:nxout,ny/2,:])
    plc(myfun[nxin:nxout,ny/2,:],xcoord=xy_x,ycoord=myy,ax=ax,cb=True,nc=50)
    ax.grid(linestyle='-')
    plt.xlabel(r"$x [r_g]$",ha='center',labelpad=0,fontsize=14)
    plt.ylabel(r"$y [r_g]$",ha='left',labelpad=20,fontsize=14)
    plt.savefig('amag'+str(fnumber)+'.png')'''

    return numtot,num_mean,num_turb, num_mix1, num_mix2, gdet

def LiTest():
    ###calculate mu
    #define a boundary of the RT bubble - just in the 2D plane?
    #mask and use r_max for bubble, r_min for high density region?
    rhoplus=np.average(rhohigh)
    rhominus=np.average(rholow)
    mu=(rhoplus-rhominus)/(rhoplus+rhominus)

    ###calculate mu1
    rm=np.average(rboundary)
    omega_m=np.sqrt(G*MBH/rm**3)
    q=1.5 #keplarian, but could be up to 2
    #how to set m? just choose?
    geff=-1/rho_0*dp_t0/dr #get initial pressure from Mark's paper
    mu1=(m*geff)/(rm*((1-q/2)*omega_m)**2)

    return mu, mu1

    ###do I really want to solve for m?


