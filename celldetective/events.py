import numpy as np

def switch_to_events(classes, times, max_times, first_detections=None, left_censored=False, FrameToMin=None):
	
	events = []
	survival_times = []
	if first_detections is None:
		first_detections = np.zeros_like(max_times)
		
	for c,t,mt,ft in zip(classes, times, max_times, first_detections):

		if left_censored:
			#print('left censored is True: exclude cells that exist in first frame')
			if ft>0.:
				if c==0:
					if t>0:
						dt = t - ft
						#print('event: dt = ',dt, t, ft)
						if dt>0:
							events.append(1)
							survival_times.append(dt)
				elif c==1:
					dt = mt - ft
					if dt>0:
						events.append(0)
						survival_times.append(dt)
				else:
					pass
		else:
			if c==0:
				if t>0:
					events.append(1)
					survival_times.append(t - ft)
			elif c==1:
				events.append(0)
				survival_times.append(mt - ft)
			else:
				pass

	if FrameToMin is not None:
		print('convert to minutes!', FrameToMin)
		survival_times = [s*FrameToMin for s in survival_times]
	return events, survival_times
	  
	