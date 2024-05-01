import unittest
from celldetective.events import switch_to_events

class TestEventSwitch(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.classes = [0,0,1,1,2]
		self.event_times = [5.,8.5,-1,-1,-1]
		self.max_times = [10,10,10,10,10]
		self.origin_times = [0,3,2,1,0]

		self.expected_events = [1,1,0,0]
		self.expected_times = [5.,5.5,8,9]

	def test_expected_events(self):
		events, times = switch_to_events(
										self.classes,
										self.event_times,
										self.max_times,
										self.origin_times
										)
		self.assertEqual(events, self.expected_events)
		self.assertEqual(times, self.expected_times)


if __name__=="__main__":
	unittest.main()