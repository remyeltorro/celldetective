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
	

	"""
	Converts time-to-event data into a format suitable for survival analysis, optionally adjusting for left censorship 
	and converting time units.

	This function processes event data by classifying each event based on whether it occurred or was censored by the end 
	of the observation period. It calculates the survival time for each event, taking into account the possibility of left 
	censorship and the option to convert time units (e.g., from frames to minutes).

	Parameters
	----------
	classes : array_like
		An array indicating the class of each event (e.g., 0 for event, 1 for non-event, 2 for else).
	event_times : array_like
		An array of times at which events occurred. For non-events, this might represent the time of last observation.
	max_times : array_like
		An array of maximum observation times for each event.
	origin_times : array_like, optional
		An array of origin times for each event. If None, origin times are assumed to be zero, and `left_censored` is 
		automatically set to False (default is None).
	left_censored : bool, optional
		Indicates whether to adjust for left censorship. If True, events with origin times are considered left-censored 
		if the origin time is zero (default is True).
	FrameToMin : float, optional
		A conversion factor to transform survival times from frames (or any other unit) to minutes. If None, no conversion 
		is applied (default is None).

	Returns
	-------
	tuple of lists
		A tuple containing two lists: `events` and `survival_times`. `events` is a list of binary indicators (1 for event 
		occurrence, 0 for censorship), and `survival_times` is a list of survival times corresponding to each event or 
		censorship.

	Notes
	-----
	- The function assumes that `classes`, `event_times`, `max_times`, and `origin_times` (if provided) are all arrays of 
	  the same length.
	- This function is particularly useful in preparing time-to-event data for survival analysis models, especially when 
	  dealing with censored data and needing to adjust time units.

	Examples
	--------
	>>> classes = [0, 1, 0]
	>>> event_times = [5, 10, 15]
	>>> max_times = [20, 20, 20]
	>>> origin_times = [0, 0, 5]
	>>> events, survival_times = switch_to_events_v2(classes, event_times, max_times, origin_times, FrameToMin=0.5)
	# This would process the events considering left censorship and convert survival times to minutes.
	
	"""

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