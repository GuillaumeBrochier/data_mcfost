import matplotlib 
import matplotlib.pyplot as plt
import pymcfost as mcfost
import numpy as np
import scipy
import numpy.ma as ma
from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
from scipy.optimize import bisect


def average(moment):
	"""
	pas ouf mais ok
	renvoie la vitesse azimuthale moyennée vue sur un disque inclinée
	"""
	x_0, y_0   = 401, 401
	r          = np.zeros(np.shape(moment))
	phi        = np.zeros(np.shape(moment))
	moment_sub = np.zeros(np.shape(moment))
	
	for i in range (0,len(moment[:,0])):
		for j in range (0, len(moment[0,:])):
			r[i,j]   = np.sqrt((i-x_0)**2/np.sin(45)**2 + (j-y_0)**2)
			try :
				phi[i,j] = np.arccos((i-x_0)/r[i,j]) 
			except:
				pass
	
	cos_phi = np.rot90(np.cos(phi))
	r_max   = np.max(r)
	ech     = 300
	v_phi   = np.zeros(ech)
	seuil   = r_max/ech
	
	moment_uni  = np.divide(moment, cos_phi, out=np.zeros_like(moment), where=np.abs(cos_phi)>0.01)
	moment_test = np.zeros(np.shape(moment))
	
	for i in range (1, ech):
		mask       = np.where(r<=(i+1)*seuil, r,1)
		mask       = np.where(mask>i*seuil, 0,1)
		v_phi[i]   = np.sum(ma.array(moment_uni, mask=mask))
		cell       = np.sum(mask)
		v_phi[i]   = v_phi[i]/cell
		#moment_i   = ma.array(moment_uni-v_phi[i], mask=mask)
		moment_i   = ma.array(moment_test+v_phi[i], mask=mask)
		moment_sub += moment_i
	
	return moment_sub*cos_phi


def lect_donnees(nom, mol="co"):
	continu    = nom+"reel_cont_"+mol
	cont_brut  = np.loadtxt(continu, unpack=True)
	
	size       = len(cont_brut[0,:])
	cont       = np.zeros((size,size))
	
	for i in range (len(cont_brut[:,0])//size):
		cont  = np.dstack((cont, np.rot90(cont_brut[size*i:size*(i+1),:])))
		
	return cont[:,:,1:]
	

def moment(co=True, c18o=True, c16o=True):
	liste = np.array([])
	liste_nom=[]
	
	if c18o:
		liste = np.append(liste, mcfost.Line("./data_C18O"))
		liste_nom.append('image_c18o')
	if c16o:
		liste = np.append(liste, mcfost.Line("./data_13C16O"))
		liste_nom.append('image_13co')
	if co:
		liste = np.append(liste, mcfost.Line("./data_CO"))
		liste_nom.append('image_co')
	
	vel = liste[0].velocity
	vel = vel[2:-2:2]
	im  = np.zeros((len(vel), 801,801))
	
	for j in range (0, len(vel)):
		print(vel[j])
		im_j = liste[0].plot_map(psf_FWHM=0.1, Delta_v=0.09, v=vel[j], i=2, substract_cont=True, cmap='nipy_spectral')
		im[j,:,:] = im_j[:,:]

	dv       = vel[1] - vel[0]
	moment_0 = np.sum(im, axis=0) * dv
	a        = np.sum(im[:, :, :] * vel[:, np.newaxis, np.newaxis], axis=0) * dv
	moment_1 = np.divide(a, moment_0, out=np.zeros_like(a), where=moment_0!=0)
	
	plt.show()
            
	return moment_0, moment_1
	
	
def casassus(moment, psi=0*np.pi/180, angle=45*np.pi/180, M=1.989E33, G=6.67E-8, au=1.496E13):
	
	def eq(y_0=10, x_0=10, psi=psi, i=angle):
		a = -np.cos(i)**2 + (np.sin(i)*np.tan(psi))**2
		b = -2*x_0*np.sin(i)*np.tan(psi)
		c = x_0**2 + y_0**2 * np.cos(i)**2
		
		delta = b**2 - 4*a*c
		sol_1 = (-b + np.sqrt(delta))/(2*a)
		sol_2 = (-b - np.sqrt(delta))/(2*a)
		#y_0**2 * np.cos(i)**2 + (x_0 - r*np.tan(psi)*np.sin(i))**2 - r**2 * np.cos(i)**2
		return sol_2
		
	x_max     = len(moment[0,:])
	y_max     = len(moment[:,0])
	r_tot     = np.zeros(np.shape(moment))
	theta_tot = np.zeros(np.shape(moment))
	
	for i in range (0,x_max):
		x_0 = i - y_max/2
		for j in range (0,y_max):
			y_0 = j - x_max/2
			
			r_sol = eq(x_0=x_0, y_0=y_0)  #np.sqrt(y_0**2+(x_0/np.sin(angle))**2)
			theta = np.arccos(y_0/r_sol)

			#x_0 = r_sol*np.sin(phi)/ np.cos(i) + (r_sol*np.tan(psi) − r_sol*np.sin(phi)*np.tan(i))*np.sin(i)
			#y_0 = r_sol*np.cos(phi)
			
			r_tot[i,j]     = r_sol
			theta_tot[i,j] = theta
			
	#plt.imshow(r_tot)
	#plt.colorbar()
	#plt.show()

	v_0 = np.sqrt(M*G/(r_tot*au)) * np.sin(angle) * np.cos(theta_tot)
	#plt.imshow(v_0/1E5, vmin=-10, vmax=10, cmap='jet')
	#plt.colorbar()
	#plt.show()
	
	ech = 400
	r   = np.copy(r_tot)
	dr  = np.max(r)/ech
	
	v_r_tot = np.zeros(np.shape(moment))
	moment  = np.where(np.abs(moment)>10, 0, moment)
	
	for i in range (0,ech):
		mini = np.where(r>i*dr, 1, 0)
		maxi = np.where(r<(i+1)*dr, 1, 0)
		aire = mini*maxi

		v_r     = np.sum(moment*aire*np.cos(theta_tot))/np.sum(aire*np.cos(theta_tot)**2)
		#v_r     = np.sum(v_0*aire*np.cos(theta_tot))/np.sum(aire*np.cos(theta_tot)**2)/1E5
		v_r_tot = v_r_tot + v_r*aire
	
	v_aver = np.cos(theta_tot) * v_r_tot
	diff   = moment - v_aver
	
	print(np.sum(np.abs(v_aver)), np.sum(np.abs(moment)))
	
	plt.figure(5)
	plt.imshow(v_aver, vmin=-10, vmax=10, cmap='bwr')
	plt.colorbar()
	plt.show()
	
	return diff
	

if __name__=="__main__":

	#mcfost.run("/home/guillaume/Documents/Stage_M1/mcfost/ref3.0_3D.para", delete_previous=True)
	
	#model = mcfost.SED("./data_th/")
	#print(model.P)
	#model.plot_T()
	#model.plot(0)

	#image_2mum = mcfost.Image("./data_1300/")
	#image_np   = image_2mum.image[0,0,0,:,:] 
	#image_co   = mcfost.Line("./data_C18O")
	#test       = image_co.lines[0,2,0,:,:,:]

	#moment_0 = image_co.get_moment_map(i=2, beam=0.1, conv_method=convolve_fft, substract_cont=True)
	#moment_1 = image_co.get_moment_map(moment=1, i=2, beam=0.1, conv_method=convolve_fft, substract_cont=True)
	
	#moment_0, moment_1 = moment(co=False, c16o=False, c18o=True)
	
	#np.savetxt("moment_0_c18o", np.column_stack((moment_0)), header = "13co 45 deg, convolue psf_FWHM=0.1, Delta_v=0.09")
	#np.savetxt("moment_1_c18o", np.column_stack((moment_1)), header = "13co 45 deg, convolue psf_FWHM=0.1, Delta_v=0.09")
	
	moment_0 = np.loadtxt("moment_0_c18o", unpack=True)
	moment_1 = np.loadtxt("moment_1_c18o", unpack=True)
	
	diff = casassus(moment_1)
	
	plt.figure(1)
	plt.imshow(diff, vmin=-0.5, vmax=0.5, cmap='bwr')
	plt.colorbar()
	#plt.show()

	fig, axes = plt.subplots(1,2)

	print(np.min(moment_0), np.max(moment_0))
	print(np.min(moment_1), np.max(moment_1))

	m_0 = axes[0].imshow(moment_0, cmap='nipy_spectral')
	fig.colorbar(m_0, ax=axes[0])
	m_1 = axes[1].imshow(moment_1, vmin=-5, vmax=5, cmap='bwr')
	fig.colorbar(m_1, ax=axes[1])

	plt.show()

