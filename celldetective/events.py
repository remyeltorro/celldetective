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
	  
	
def switch_to_events_v2(classes, event_times, max_times, origin_times=None, left_censored=True, FrameToMin=None):
	
	events = []
	survival_times = []

	if origin_times is None:
		# then origin is zero
		origin_times = np.zeros_like(max_times)
		left_censored = False
		
	for c,t,mt,ot in zip(classes, event_times, max_times, origin_times):

		if left_censored:

			if ot>=0. and ot==ot:
				# origin time is larger than zero, no censorship
				if c==0 and t>0:
					delta_t = t - ot
					if delta_t>0:
						events.append(1)
						survival_times.append(delta_t)
					else:
						# negative delta t, invalid cell
						pass
				elif c==1:
					delta_t = mt - ot
					if delta_t>0:
						events.append(0)
						survival_times.append(delta_t)
					else:
						# negative delta t, invalid cell
						pass
				else:
					pass
			else:
				# origin time is zero, the event is left censored (we did not observe it start)
				pass

		else:
			if c==0 and t>0:
				events.append(1)
				survival_times.append(t - ot)
			elif c==1:
				events.append(0)
				survival_times.append(mt - ot)
			else:
				pass

	if FrameToMin is not None:
		print('convert to minutes!', FrameToMin)
		survival_times = [s*FrameToMin for s in survival_times]
	return events, survival_times