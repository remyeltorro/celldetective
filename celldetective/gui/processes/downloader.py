import os
import ssl
from tqdm import tqdm
from multiprocessing import Process
from glob import glob
import shutil
from urllib.request import urlopen
import zipfile
import tempfile
import time

class DownloadProcess(Process):

	def __init__(self, queue=None, parent_window=None):
		
		super().__init__()
		
		self.queue = queue
		self.parent_window = parent_window
		self.output_dir = self.parent_window.output_dir
		self.file = self.parent_window.file

		zenodo_json = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "links", "zenodo.json"])
		print(f"{zenodo_json=}")
		
		with open(zenodo_json,"r") as f:
			zenodo_json = json.load(f)
		all_files = list(zenodo_json['files']['entries'].keys())
		all_files_short = [f.replace(".zip","") for f in all_files]
		zenodo_url = zenodo_json['links']['files'].replace('api/','')
		full_links = ["/".join([zenodo_url, f]) for f in all_files]
		index = all_files_short.index(file)
		
		self.zip_url = full_links[index]
		self.path_to_zip_file = os.sep.join([self.output_dir, 'temp.zip'])

		self.sum_done = 0
		self.t0 = time.time()

	def download_url_to_file(self, url, dst):

		file_size = None
		ssl._create_default_https_context = ssl._create_unverified_context
		u = urlopen(url)
		meta = u.info()
		if hasattr(meta, 'getheaders'):
			content_length = meta.getheaders("Content-Length")
		else:
			content_length = meta.get_all("Content-Length")
		if content_length is not None and len(content_length) > 0:
			file_size = int(content_length[0])
		# We deliberately save it in a temp file and move it after
		dst = os.path.expanduser(dst)
		dst_dir = os.path.dirname(dst)
		f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
		try:
			with tqdm(total=file_size, disable=not progress,
					  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
				while True:
					buffer = u.read(8192) #8192
					if len(buffer) == 0:
						break
					f.write(buffer)
					pbar.update(len(buffer))
					print(f"{len(buffer)=}")
					self.queue.put(len(buffer))
			f.close()
			shutil.move(f.name, dst)
		finally:
			f.close()
			if os.path.exists(f.name):
				os.remove(f.name)

	def run(self):
		
		self.download_url_to_file(fr"{self.zip_url}",self.path_to_zip_file)
		with zipfile.ZipFile(self.path_to_zip_file, 'r') as zip_ref:
			zip_ref.extractall(self.output_dir)
		
		file_to_rename = glob(os.sep.join([self.output_dir,self.file,"*[!.json][!.png][!.h5][!.csv][!.npy][!.tif][!.ini]"]))
		if len(file_to_rename)>0 and not file_to_rename[0].endswith(os.sep) and not self.file.startswith('demo'):
			os.rename(file_to_rename[0], os.sep.join([self.output_dir,self.file,self.file]))

		os.remove(self.path_to_zip_file)
		self.queue.put(100)
		time.sleep(1)

		# Send end signal
		self.queue.put("finished")
		self.queue.close()

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")