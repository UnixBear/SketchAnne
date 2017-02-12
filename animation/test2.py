import glob
from natsort import natsorted, ns
filelist = natsorted([filename for filename in glob.iglob('/home/morpheus/Projects/SketchAnne/animation/*.png')])
print filelist
