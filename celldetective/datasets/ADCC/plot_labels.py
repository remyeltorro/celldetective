import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tifffile import imread
from natsort import natsorted
import os 
from matplotlib import patches
import shutil

label_path = natsorted(glob('mcf7_nuclei_w_primary_NK/*_labelled.tif'))
labels = [imread(f) for f in label_path]
images = [imread(f.replace('_labelled.tif','.tif'))[0] for f in label_path]

if os.path.exists('control'+os.sep):
	shutil.rmtree('control'+os.sep)
os.mkdir('control'+os.sep)

#end()
X = np.array(images,dtype=int)
Y = np.array(labels,dtype=int)
#X = np.moveaxis(X,1,-1)

print(np.shape(X))

idx = 0
for k in range(len(X)):
	fig,ax = plt.subplots(1,2,figsize=(5,3))
	ax[0].imshow(X[k,:,:],alpha=1,cmap="gray")
	#plt.imshow(X[k,:,:,0],alpha=0.5,cmap="gray")
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	ax[1].imshow(Y[k],cmap="Blues")
	#ax[1].imshow(X[k,:,:,0],alpha=1,cmap="gray")

	unique_cells = np.unique(Y[k])
	for c in unique_cells[1:]:
		idx+=1
		indices_i,indices_j = np.where(Y[k]==c)
		ymin, ymax, xmin, xmax = np.amin(indices_i),np.amax(indices_i),np.amin(indices_j),np.amax(indices_j)
		w = xmax - xmin
		h = ymax - ymin
		box = patches.Rectangle(
			(xmin, ymin), w, h, edgecolor="red", facecolor="none"
		)

		ax[1].add_patch(box)
		ax[1].scatter([np.mean(indices_j)],[np.mean(indices_i)],marker="x",s=2,color="tab:red")

	ax[1].set_xticks([])
	ax[1].set_yticks([])
	name = os.path.split(label_path[k])[-1]
	plt.savefig(f"control/{name[:-4]}.png",dpi=300,bbox_inches="tight")
	plt.pause(0.5)
	plt.close()
	
print(idx)
