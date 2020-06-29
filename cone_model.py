import matplotlib 
import matplotlib.pyplot as plt
import pymcfost as mcfost
import numpy as np
import scipy
from scipy.optimize import curve_fit, bisect, fmin, fmin_powell
from astropy.convolution import convolve, Gaussian1DKernel
from matplotlib.patches import Ellipse
from pynverse import inversefunc


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
		
	vel = liste[0].velocity[140:261:2]
	print(vel)
	
	for i in range(0,len(liste_nom)):
		im=np.zeros((len(vel),401,401))
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
			
		print(np.shape(im), np.shape(liste[0].lines[0,2,0,140:261:2,::2,::2]))
		plt.show()
	
	#np.savetxt(simu+"donnees_co", np.column_stack((liste[0].lines[0,2,0,::20,::2,::2])), header = "i=45 deg")
	np.savetxt(simu+"reel_cont_13co", np.column_stack((im)), header = "continu 45 deg, convolue psf_FWHM=0.1, Delta_v=0.09")

	#np.savetxt(simu+"vel_reel", np.column_stack((vel)), header="velocite des channels")
	
	return


def lect_donnees(simu, nom, mol="co"):
	path       = nom
	continu    = path+"reel_cont_"+mol
	cont_brut  = np.loadtxt(continu, unpack=True)
	
	size       = len(cont_brut[0,:])
	cont       = np.zeros((size,size))
	
	for i in range (len(cont_brut[:,0])//size):
		cont  = np.dstack((cont, np.rot90(cont_brut[size*i:size*(i+1),:])))
		
	return cont[:,:,1:]
	
	
def v_kepler(r,h, G=6.67E-8, M=2.4*1.989E33, au=1.496E13):
	#le facteur 1000/401 convertit les pixels en au, pour obtenir une vitesse en cm/s (1000 = size, dans fichier para)
	r = au*r*1000/401
	h = au*h*1000/401
	return np.sqrt(M*G*r**2/np.abs((r**2+h**2)**(3/2)))
	
	
def fit_theo(vit, angle=45*np.pi/180, G=6.67E-8, M=2.4*1.989E33, au=1.496E13):
	theta = np.linspace(-np.pi, np.pi, 600)

	r = G*M/(vit*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2*401/(1000*au)
	x = r*np.cos(theta) + 201
	y = r*np.sin(theta) + 201
	
	return x, y, theta


def fit_pos(x, y, t, angle=45*np.pi/180, G=6.67E-8, M=2.4*1.989E33):
	x = x-201
	y = y-201
	#r = np.sqrt(x**2 + y**2)
	#v = np.sqrt(G*M /r) *np.sin(angle)
	
	im = np.zeros((len(x),len(y)))
	for i in range (0, len(im[:,0])):
		for j in range (0, len(im[0,:])):
			r = np.sqrt(x[j]**2 + (y[i]-t[i,j] *np.sin(angle))**2) 
			v = np.sqrt(G*M /r) *np.sin(angle)
			#le moins est conventionnel, à cause de l'axe imshow inversé
			im[i,j] = -v*x[j]/r
			
	return im

	
def eq(t, angle=45.*np.pi/180, psi=15.*np.pi/180, x=10., y=10.):
	x = x-201
	y = y-201
	#y = (y+t*np.sin(i))*np.cos(angle)
	return t**2 *(np.cos(2*angle) + np.cos(2*psi))- 2* np.sin(psi)**2 *(x**2 + y**2/np.cos(angle)**2 + 2*t*y*np.tan(angle))


def sol(x, y, psi=15*np.pi/180, angle=45*np.pi/180):
	b = 4* np.sin(psi)**2* y*np.sin(angle)
	a = np.cos(2*angle) + np.cos(2*psi)
	c = -2* np.sin(psi)**2 *(x**2 + y**2)
	delta = b**2 -4*a*c
	
	if delta >=0:
		sol_1 = (-b + np.sqrt(delta))/(2*a)
		sol_2 = (-b - np.sqrt(delta))/(2*a)
	else:
		print(delta, a,b,c)
		sol_1, sol_2 = 0, 0
		
	return sol_1, sol_2


def fit_image(para, t=np.zeros((401,401)),  im=np.zeros((401,401)), vit=-1, print_tot=True, delta_v=0.05, angle=45*np.pi/180, G=6.67E-8, M=2.4*1.989E33):
	"""
	Attention ! changer selon les paramètres des images : nombre d'au (ref3.0_2.para), de pixels de l'image notamment
	"""
	vit   = para[0]
	a, b  = 0, 0
	theta = np.linspace(-np.pi, np.pi, 600)
	
	if a<0 or b<0:
		print('non physique !')
		return 0
	
	mass = (lambda x: x/(1+a*x**b))
	inv  = inversefunc(mass, domain=[0, 1E4], image=[0,1E4])
	
	r_mod = G*M/((vit+delta_v)*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2 *401/(1000*1.496E13)
	#r_mod = inv(G*M/((vit+delta_v)*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2 *401/(1000*1.496E13))
	#r_mod = (1+a*r**b)*r
	
	x_plus, y_plus   = r_mod*np.cos(theta) + 201, r_mod*np.sin(theta) + 201
	
	r_mod = G*M/((vit-delta_v)*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2 *401/(1000*1.496E13)
	#r_mod = inv(G*M/((vit-delta_v)*1E5)**2 * np.sin(angle)**2 * np.cos(theta)**2 *401/(1000*1.496E13))
	#r_mod = (1+a*r**b)*r
	
	x_moins, y_moins = r_mod*np.cos(theta) + 201, r_mod*np.sin(theta) + 201
	
	#print('ok')
	
	aire = np.zeros(np.shape(im))
	for i in range(0, len(aire[:,0])):
		y = i - 201
		for j in range(0, len(aire[0,:])):
			x = j - 201
			if x != 0: 
				theta_ij = np.arctan(y/x)
			else:
				theta_ij = 0
			theta_fit = np.sum(np.where(theta<theta_ij, 1,0))
			
			r_ij  = x**2 + y**2
			plus  = (x_plus[theta_fit]-201)**2 + (y_plus[theta_fit]-201 - t[i,j]*np.sin(angle))**2
			moins = (x_moins[theta_fit]-201)**2 + (y_moins[theta_fit]-201- t[i,j]*np.sin(angle))**2
			
			#if i%10==0 and j%10==0:
			#	print(r_ij, plus, moins)
			#	plt.plot(x_plus, y_plus)
			#	plt.plot(x_moins, y_moins)
			#	plt.plot(x+201, y+201, 'o')
			#	plt.show()
			R_out = 200**2
			moins, plus = min(moins, plus), max(moins, plus)
			if r_ij  > moins and r_ij < min(plus, R_out) :#and x<=0:
				aire[i,j] = 1
				
	im_mod = im*aire
	tot    = -np.sum(im_mod)/np.sum(aire)
	
	if print_tot:
		return tot
	else:
		return im_mod


def maxima(continu):

	longueur = len(continu[:,0,0])
	hauteur  = len(continu[0,:,0])
	nb_lines = len(continu[0,0,:])
	max_up   = np.zeros((nb_lines, longueur))
	max_down = np.zeros((nb_lines, longueur))

	x_up   = np.linspace(hauteur//2, hauteur, len(continu[hauteur//2:,0,0]))
	x_down = np.linspace(0, hauteur//2, len(continu[hauteur//2:,0,0]))
	
	for i in range (0, nb_lines):
		for j in range (0,longueur):
			line_up     = continu[hauteur//2:,j,i]
			max_up[i,j] = hauteur//2+np.argmax(line_up)
			
			line_down     = continu[:hauteur//2,j,i]
			max_down[i,j] = np.argmax(line_down)
	
	return max_up, max_down
	

def position(max_up, max_down, i, vit, x=np.linspace(0,200,200)):
	x_star   = 201
	y_star   = x_star
	
	x_milieu = x_star
	y_milieu = (max_up + max_down)/2
	
	r        = np.sqrt((x-x_star)**2 + (max_up - y_milieu)**2/(np.cos(i)**2))
	h        = (y_milieu - y_star)/np.sin(i)
	vit_reel = np.abs(1*vit*1E5 * r / ((x-x_star)*np.sin(i)))
	
	return r, np.abs(h), vit_reel
	

def v_laibe(r, h, p=1, q=0.25, G=6.67E-8, M=2.4*1.989E33, au=1.496E13):
	#la valeur de c_s est prise dans le fichier discparam de phantom, le facteur 1E5 permet de passer en cm/s
	c_s = 0.03696 *1E5* r**(-q/2)
	r   = r*1000/401*au
	h   = h*1000/401*au
	v   = np.sqrt(G*M/r -(p + q/2 + 1.5)*c_s**2 *r**(-q) - G*M*q*(1/r - 1/np.sqrt(r**2 + h**2)))
	
	return v


if __name__=="__main__":

	simu    = "beta_12_rad_250"
	angle   = 45*np.pi/180
	psi     = 1*np.pi/180
	vit     = np.linspace(-3,3,61)
	channel = 40
	print(vit[channel])
	#import_donnees(simu, co=False, c18o=False, c16o=True)
	
	nom     = "beta_12_rad_250"
	cont_co = lect_donnees(simu, nom, mol="co")
	cont_13 = lect_donnees(simu, nom, mol="13co")
	cont_18 = lect_donnees(simu, nom, mol="c18o")
	
	cont = cont_13
	
	up, down         = maxima(cont)
	gauss            = Gaussian1DKernel(5)
	max_up, max_down = up[channel,:200], down[channel,:200]
	max_up, max_down = convolve(np.concatenate(([max_up[0]]*10, max_up, [max_up[-1]]*10)), gauss)[10:-10], convolve(np.concatenate(([max_down[0]]*10, max_down, [max_down[-1]]*10)), gauss)[10:-10]
	
	t_1 = np.zeros(np.shape(cont[:,:,channel]))
	t_2 = np.zeros(np.shape(cont[:,:,channel]))	
	
	for x in range (0,len(t_1[:,0])):
		for y in range (0, len(t_1[0,:])):
			t_1[y, x], t_2[y, x] = sol(x-201, y-201, psi=psi) 

	#fit des données avec "direct mapping ..."
	r, h, vit_reel = position(max_up, max_down, angle, vit[channel])
	
	plt.figure(1)
	plt.imshow(cont[:,:,channel], cmap='inferno')
	plt.plot(max_up)
	plt.plot(max_down)

	plt.figure(2)
	plt.plot(r, vit_reel, 'o', label='vitesse reelle')
	plt.plot(r, v_laibe(r, h), '*', label='vitesse de laibe')
	plt.plot(r, v_kepler(r,h), '-', label='vitesse de kepler')
	plt.legend()
	
	#plt.figure(3)
	#plt.plot(v_kepler(r, h)/vit_reel, '-', label='vitesse de kepler')
	#plt.legend()
	plt.show()
	
	
	#fit des données avec un profil théorique des channels
	#mini = fmin(fit_image, [vit[channel]], args=(t_2, cont[:,:,channel], vit[channel]))
	#print(mini)
	
	im_tot = np.zeros(np.shape(cont[:,:,channel]))
	for i in range(0,len(vit)):
		vel    = vit[i]
		#mini = fmin(fit_image, [vel], args=(t_2, cont[:,:,i], vit[i]))
		im_tot = im_tot + fit_image([vel], t=t_2, im=cont[:,:,i], vit=vel, print_tot=False)
		print(vel)
	
	plt.figure(1)
	plt.imshow(im_tot, cmap='inferno')
	plt.colorbar()
	plt.show()
	
	#plt.figure(1)
	#plt.imshow(cont[:,:,channel]-fit_image([vit[channel]+0.3], t=t_2, im=cont[:,:,channel], vit=vit[channel], print_tot=False), cmap='inferno')
	#plt.colorbar()
	
	#plt.figure(2)
	#plt.imshow(cont[:,:,channel], cmap='inferno')
	#plt.colorbar()
	
	#plt.show()


	#reproduction théorique du modèle du double cone
	#plt.figure(1)
	#plt.imshow(t_1, cmap='jet')
	#plt.imshow(fit_pos(np.linspace(0,401,401),np.linspace(0,401,401), t_1), vmin=-3e12, vmax=3e12, cmap ='jet')
	#plt.colorbar()
	
	#plt.figure(2)
	#plt.imshow(t_2, cmap='jet')
	#plt.imshow(fit_pos(np.linspace(0,401,401),np.linspace(0,401,401), t_2), vmin=-3e12, vmax=3e12, cmap ='jet')
	#plt.colorbar()
	
	#plt.show()		




