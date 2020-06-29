import matplotlib 
import matplotlib.pyplot as plt
import pymcfost as mcfost
import numpy as np
import scipy
from scipy.optimize import curve_fit 
from astropy.convolution import convolve, Gaussian1DKernel
from matplotlib.patches import Ellipse


def fitEllipse(contour):

    x = contour[:,0]
    y = contour[:,1]
    x = x[:,None]
    y = y[:,None]

    D = np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V   = np.linalg.eig(np.dot(np.linalg.inv(S),C))

    n = np.argmax(E)
    a = V[:,n]

	#fit ellipse
    b,c,d,f,g,a = a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num = b*b-a*c
    cx  = (c*d-b*f)/num
    cy  = (a*f-b*d)/num

    angle = 0.5*np.arctan(2*b/(a-c))*180/np.pi
    up    = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a     = np.sqrt(abs(up/down1))
    b     = np.sqrt(abs(up/down2))

    ell       = Ellipse((cx,cy),a*2.,b*2.,angle)
    ell_coord = ell.get_verts()
    params    = [cx,cy,a,b,angle]

    return params,ell_coord


def import_donnees(simu, co=True, c18o=True, c16o=True):
	liste     = np.array([])
	liste_nom = []
	
	if co:
		liste = np.append(liste, mcfost.Line("./data_CO"))
		liste_nom.append('image_co')
	if c18o:
		liste = np.append(liste, mcfost.Line("./data_C18O"))
		liste_nom.append('image_c18o')
	if c16o:
		liste = np.append(liste, mcfost.Line("./data_13C16O"))
		liste_nom.append('image_13co')
		
	vel = liste[0].velocity[140:221:2]
	print(vel)
	
	im=np.zeros((len(vel),401,401))
	
	for i in range(0,len(liste_nom)):
		#fig, axes = plt.subplots(1,1)
		for j in range (0,len(vel)) :
			no_ylabel = False
			colorbar  = False
			if j > 0 :
				no_ylabel = True
			if j==2 :
				colorbar  = True
			im_j = liste[i].plot_map(Tb=True, title=liste_nom[i],psf_FWHM=0.1, Delta_v=0.09, v=vel[j], i=2, substract_cont=True, no_ylabel = no_ylabel, colorbar = colorbar, cmap='nipy_spectral', plot_stars=False)
			im[j,:,:] = im_j[::2,::2]
			plt.imshow(im_j)
			plt.show()
			
	print(np.shape(im), np.shape(liste[0].lines[0,2,0,140:221:2,::2,::2]))
	plt.show()
	
	#np.savetxt(simu+"donnees_co", np.column_stack((liste[0].lines[0,2,0,::20,::2,::2])), header = "i=45 deg")
	#np.savetxt(simu+"reel_cont_13co", np.column_stack((im)), header = "continu 45 deg, convolue psf_FWHM=0.1, Delta_v=0.09")
	#np.savetxt(simu+"vel_reel", np.column_stack((vel)), header="velocite des channels")
	
	return


def lect_donnees(simu, nom, mol="co"):
	path       = nom
	continu    = path+"reel_cont_"+mol
	cont_brut  = np.loadtxt(continu, unpack=True)
	
	size       = len(cont_brut[0,:])
	lines      = np.zeros((size,size))
	cont       = lines
	
	for i in range (len(cont_brut[:,0])//size):
		cont  = np.dstack((cont, np.rot90(cont_brut[size*i:size*(i+1),:])))
		
	return cont[:,:,1:]
	
	
def maxima(continu):

	longueur = len(continu[:,0,0])
	hauteur  = len(continu[0,:,0])
	nb_lines = len(continu[0,0,:])
	max_up   = np.zeros((nb_lines, longueur))
	max_down = np.zeros((nb_lines, longueur))

	#lines  = np.maximum(lines - continu[:,:,:], 0.0)
	x_up   = np.linspace(hauteur//2, hauteur, len(continu[hauteur//2:,0,0]))
	x_down = np.linspace(0, hauteur//2, len(continu[hauteur//2:,0,0]))
	
	for i in range (0, nb_lines):
		for j in range (0,longueur):
			try:
				line_up     = continu[hauteur//2:,j,i]
				#popt, pcov  = curve_fit(gaussienne, x_up, line_up, p0 = [np.max(line_up), hauteur//2+np.argmax(line_up), 10])
				max_up[i,j] = hauteur//2+np.argmax(line_up)
			except:
				max_up[i,j]   = hauteur//2

			try:
				line_down     = continu[:hauteur//2,j,i]
				#popt, pcov    = curve_fit(gaussienne, x_up, line_up, p0 = [np.max(line_down), np.argmax(line_down), 10])
				max_down[i,j] = np.argmax(line_down)
			except:
				max_down[i,j] = hauteur//2
				
	return max_up, max_down
	
	
def gaussienne(x,maxi, mean, std):
	return maxi*np.exp(-1*(x-mean)**2/std**2)
	

def position(max_up, max_down, i, vit, x=np.linspace(201,402,201)):
	x_star   = 201
	y_star   = x_star
	
	x_milieu = x_star
	y_milieu = (max_up + max_down)/2
	
	r = np.sqrt((x-x_star)**2 + (max_up - max_down)**2/(4*np.cos(i)**2))
	h = (max_down + max_up - 2*y_star)/(2*np.sin(i))
	vit_reel = np.abs(vit*1E5 * r / ((x-x_star)*np.sin(i)))
	
	return r, h, vit_reel
	
	
def position_ell(para, i, vit):
	x_star   = 201
	y_star   = 201
	x_milieu = x_star
	
	#on reconstruit l'ellipse
	cx, cy, a, b, psi = para[0], para[1], para[2], para[3], para[4]
	
	ech   = 400
	theta = np.linspace(0,2*np.pi, ech)
	x_pos = a*np.cos(theta)
	y_pos = b*np.sin(theta)

	new_xpos =  x_pos*np.cos(psi) + y_pos*np.sin(psi) + cx
	new_ypos = -x_pos*np.sin(psi) + y_pos*np.cos(psi) + cy
	
	x     = np.linspace(np.min(new_xpos), np.max(new_xpos), ech//2)
	x_min = np.min(new_xpos)
	
	y_milieu = np.array([])
	r        = np.array([])
	h        = np.array([])
	vit_reel = np.array([])
	
	x_up   = np.where(new_ypos>201, new_xpos, 0)
	x_down = np.where(new_ypos<201, new_xpos, 0)
	
	x_up_z   = np.sum(np.where(x_up==0, 1, 0))
	x_down_z = np.sum(np.where(x_down==0, 1, 0))
		
	for absc in x :
		if absc <0:
			absc_up   = np.sum(np.where(x_up<absc, 1,0)) 
			absc_down = np.sum(np.where(x_down<absc, 1,0))
		else : 
			absc_up   = np.sum(np.where(x_up<absc, 1,0)) - x_up_z
			absc_down = np.sum(np.where(x_down<absc, 1,0)) - x_down_z
		try :
			max_up    = new_ypos[absc_up]
			max_down  = new_ypos[absc_down]
			y_milieu = np.append(y_milieu, (max_up + max_down)/2) 
			r        = np.append(r, np.sqrt((absc-x_star)**2 + (max_up - max_down)**2/np.cos(i)**2))
			h        = np.append(h, (max_down + max_up - 2*y_star)/(2*np.sin(i)))
			vit_reel = np.append(vit_reel, np.abs(vit*1E5 * r[-1] / (int(np.argwhere(x==absc))-x_star)*np.sin(i) ))
		except:
			pass
		
	return r, h, vit_reel
	
	
def v_kepler(r):
	#le facteur 500/401 convertit les pixels en au, pour obtenir une vitesse en cm/s (500 = size, dans fichier para)
	return np.sqrt(1.989E33*6.67E-8/np.abs(1.496E13*r*500/401))
	
	
def fit_theo(vit, angle=45):
	theta = np.linspace(0, 2*np.pi, 200)
	r = 6.67E-8*1.989E33/(vit*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2*401/(500*1.496E13)
	
	x = r*np.cos(theta) + 201
	y = r*np.sin(theta) + 201
	
	return x, y
	

if __name__=="__main__":

	simu    = "beta_12_rad_250"
	angle   = 45
	vit     = np.linspace(-5,-1,41)
	channel = 40
	#import_donnees(simu, co=False, c18o=False, c16o=True)

	nom              = "beta_12_rad_250"
	cont             = lect_donnees(simu, nom, mol="13co")
	max_up, max_down = maxima(cont)
	
	for i in range (0, len(vit)):
		x, y = fit_theo(vit[i])
		plt.imshow(cont[:,:,channel], cmap='inferno')
		plt.plot(x,y)
	plt.show()
	
	len_max   = len(max_up[0,:])
	kernel    = Gaussian1DKernel(5)
	up_mod    = np.concatenate((max_up[channel,:], [max_up[channel,-1]]*30))
	down_mod  = np.concatenate((max_down[channel,:], [max_down[channel,-1]]*30))
	up_conv   = convolve(up_mod, kernel)[:-30]
	down_conv = convolve(down_mod, kernel)[:-30]
	
	#tout ce qui concerne le fit elliptique

	contour_up       = np.array([(i, up_conv[i]) for i in range (len_max//2, len_max)])
	contour_down     = np.array([(i, down_conv[i]) for i in range (len_max//2, len_max)])
	para, ell        = fitEllipse(np.concatenate((contour_up, contour_down)))
	
	plt.imshow(cont[:,:,channel], cmap='inferno')
	plt.plot(up_conv)
	plt.plot(down_conv)
	plt.plot(ell[:,0], ell[:,1])
	plt.show()
	
	print(Ellipse((para[0],para[1]), para[2], para[3], para[4]))
	r, h, vit_reel = position(up_conv[200:], down_conv[200:], angle, vit[channel])
	
	plt.figure(1)
	plt.plot(vit_reel, 'o', label='vit_reel')
	plt.plot(v_kepler(r), 'o', label='v_k')
	plt.legend()
	plt.show()
	
	
	#le fit initial avec l'image
	#for i in range (0,len(vit)):
	#	x = np.linspace(0,len(max_up[i,:]), len(max_up[i,:]))
	#	r , h, vit_reel  = position(x, max_up[i,:], max_down[i,:],angle, vit[i])
	#	plt.figure(2*i)
	#	plt.imshow(cont[:,:,i], cmap='nipy_spectral')
	#	plt.plot(max_up[i,:])
	#	plt.plot(max_down[i,:])
		
	#	plt.figure(2*i+1)
	#	#plt.plot(r)
	#	#plt.plot(h)
	#	plt.plot(vit_reel)
	#	plt.plot(v_kepler(r))
	#	plt.show()
	
	
	
	
