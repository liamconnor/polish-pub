import matplotlib.pylab as plt
import numpy as np

from astropy.modeling.models import Sersic2D

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

class SimRadioGal:

    def __init__(self, 
                nx=2000,
                ny=2000, 
                pixel_size=0.25,
                nchan=1,
                src_density_sqdeg=13000):
        self._nx = nx
        self._ny = ny
        self._pixel_size = pixel_size
        self._nchan = nchan
        self._src_density_sqdeg = src_density_sqdeg


    def galparams(self):
        # Choose random uniform coordinates
        self.xind = np.random.randint(0, self._nx)
        self.yind = np.random.randint(0, self._ny)

        # Assume broken powerlaw source counts 
        nfluxhigh = np.random.uniform(0,0.1)**(-2./3.)
        nfluxlow = np.random.uniform(0.05,1)**(-1.)
        self.flux = nfluxhigh*nfluxlow

        # Galaxy size (sigma)
        self.sigx = np.random.gamma(2.25,1.5) * 0.5 / self._pixel_size

        # Simulate ellipticity as Tunbridge et al. 2016
        self.ellipticity=np.random.beta(1.7,4.5)

        self.sigy = self.sigx * ((1-self.ellipticity)/(1+self.ellipticity))**0.5

        self.coords = np.meshgrid(np.arange(0, 150), np.arange(0, 150))
        self.rho = np.random.uniform(-90,90)

    def gaussian2D(self, 
                  coords=None,  # x and y coordinates for each image.
                  amplitude=1,  # Highest intensity in image.
                  xo=75,  # x-coordinate of peak centre.
                  yo=75,  # y-coordinate of peak centre.
                  sigma_x=3,  # Standard deviation in x.
                  sigma_y=3,  # Standard deviation in y.
                  rho=0,  # Correlation coefficient.
                  offset=0,
                  rot=0):  # rotation in degrees.
        if coords is None:
            self.galparams()
            coords = self.coords

        x, y = coords

        rot = np.deg2rad(rot)

        x_ = np.cos(rot)*x - y*np.sin(rot)
        y_ = np.sin(rot)*x + np.cos(rot)*y

        xo = float(xo)
        yo = float(yo)

        xo_ = np.cos(rot)*xo - yo*np.sin(rot) 
        yo_ = np.sin(rot)*xo + np.cos(rot)*yo

        x,y,xo,yo = x_,y_,xo_,yo_

        # Create covariance matrix
        mat_cov = [[sigma_x**2, rho * sigma_x * sigma_y],
                   [rho * sigma_x * sigma_y, sigma_y**2]]
        mat_cov = np.asarray(mat_cov)
        # Find its inverse
        mat_cov_inv = np.linalg.inv(mat_cov)

        # PB We stack the coordinates along the last axis
        mat_coords = np.stack((x - xo, y - yo), axis=-1)

        G = amplitude * np.exp(-0.5*np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                        mat_cov_inv),
                                              mat_coords[..., np.newaxis])) + offset
        return G.squeeze()

    def sersic2d(self, 
                 coords,  # x and y coordinates for each image.
                 amplitude=1,  # Highest intensity in image.
                 xo=75,  # x-coordinate of peak centre.
                 yo=75,  # y-coordinate of peak centre.
                 sigma_x=1,  # Standard deviation in x.
                 sigma_y=1,  # Standard deviation in y.
                 rho=0,  # Correlation coefficient.
                 ellipticity=0,
                 rot=0):  # rotation in degrees.
        mod = Sersic2D(amplitude=amplitude, r_eff=25, n=4, x_0=xo, y_0=yo, 
                       ellip=ellipticity, theta=np.deg2rad(rot))

        x,y = coords

        return mod(x,y)


    def distort_galaxy(self, 
                       gal_arr, 
                       alpha=20.0):
        gal_arr = gal_arr[:,:,None]*np.ones([1,1,3])
        gal_arr_distort = elastic_transform(gal_arr, alpha=alpha,
                                           sigma=3, alpha_affine=0)[:,:,0]
        return gal_arr_distort


    def get_coords(self, xind, yind, data):
        xmin, xmax = max(0,xind-150//2), min(xind+150//2,data.shape[0])
        ymin, ymax = max(0,yind-150//2), min(yind+150//2,data.shape[1])

        return xmin, xmax, ymin, ymax 

    def sim_sky(self, nsrc=None, noise=True, 
                background=False, fnblobout=None, 
                nchan=1, distort_gal=False):
        nx, ny = self._nx, self._ny
        data = np.zeros([nx, ny, nchan])
        
        if nsrc is None:
            nsrc_ = self._src_density_sqdeg*(nx*ny*self._pixel_size**2/(3600.**2))
            nsrc = np.random.poisson(int(nsrc_))

        print("Simulating %d sources" % nsrc)

        if background:
            pass

        if fnblobout is not None:
            f = open(fnblobout,'a+')
            f.write('# xind  yind  sigx  sigy  orientation  flux\n')

        for ii in range(nsrc):
            self.galparams()
            source_ii = self.gaussian2D(self.coords,
                                   amplitude=self.flux,
                                   xo=150//2,
                                   yo=150//2,
                                   sigma_x=self.sigx,
                                   sigma_y=self.sigy,
                                   rot=self.rho,
                                   offset=0)

            if distort_gal:
                source_ii = self.distort_galaxy(source_ii, 
                                                alpha=50.0)

            xmin, xmax, ymin, ymax = self.get_coords(self.xind, self.yind, data)

            if nchan==1:
                data[xmin:xmax, ymin:ymax, 0] += (source_ii.T)[\
                            abs(min(0, self.xind-150//2)):min(150, 150+nx-(self.xind+150//2)),\
                            abs(min(0, self.yind-150//2)):min(150, 150+ny-(self.yind+150//2))]
            else:
                spec_ind = np.random.normal(0.55,0.25)
                for nu in range(nchan):
                    Snu = (source_ii.T)[\
                                abs(min(0, self.xind-150//2)):min(150, 150+nx-(self.xind+150//2)),\
                                abs(min(0, self.yind-150//2)):min(150, 150+ny-(self.yind+150//2))]
                    Snu *= (freqarr[nu]/1.4)**(-spec_ind)
                    data[xmin:xmax, ymin:ymax, nu] += Snu

        return data














