import Tkinter as tk
from Tkinter import *
import logging
import os
import platform
import sys
from collections import deque
from itertools import cycle
from PIL import Image  
from PIL import ImageTk
import psutil

class Slideshow(object):
    def __init__(self, parent, filenames, slideshow_delay=2, history_size=200):
        self.ma = parent.winfo_toplevel()
        self.filenames = cycle(filenames)
        self._files = deque(maxlen=history_size)
        self._photoimage = None
        self._id = None
        self.imglbl = tk.Label(parent)
        self.imglbl.pack(fill=tk.BOTH, expand=True)
        self.imglbl.after(1, self._slideshow, slideshow_delay * 1000)

    def _slideshow(self, delay_milliseconds):
        self._files.append(next(self.filenames))
        self.show_image()
        self.imglbl.after(delay_milliseconds, self._slideshow, delay_milliseconds)

    def show_image(self):
        filename = self._files[-1]
        image = Image.open(filename) 

        w, h = self.ma.winfo_width(), self.ma.winfo_height()
        if image.size[0] > w or image.size[1] > h:

            if w < 3 or h < 3:  # too small
                return  # do nothing
            image.thumbnail((w - 2, h - 2), Image.ANTIALIAS)

        self._photo_image = ImageTk.PhotoImage(image)
        self.imglbl.configure(image=self._photo_image)

        self.ma.wm_title(filename)

    def _show_image_on_next_tick(self):
        if self._id is not None:
            self.imglbl.after_cancel(self._id)
        self._id = self.imglbl.after(1, self.show_image)

    def next_image(self, event_unused=None):
        self._files.rotate(-1)
        self._show_image_on_next_tick()

    def prev_image(self, event_unused=None):
        self._files.rotate()
        self._show_image_on_next_tick()

    def fit_image(self, event=None, _last=[None] * 2):
        if event is not None and event.widget is self.ma and (
                _last[0] != event.width or _last[1] != event.height):
            _last[:] = event.width, event.height
            self._show_image_on_next_tick()


def get_image_files(rootdir):
    for path, dirs, files in os.walk(rootdir):
        dirs.sort()  
        files.sort()  
        for filename in files:
            if filename.lower().endswith('.jpg'):
                yield os.path.join(path, filename)


def main():
    logging.basicConfig(format="%(asctime)-15s %(message)s",
                        datefmt="%F %T",
                        level=logging.DEBUG)

    root = tk.Tk()

    imagedir = sys.argv[1] if len(sys.argv) > 1 else '.'
    image_filenames = get_image_files(imagedir)

    if platform.system() == "Windows":
        root.wm_state('zoomed')  # start maximized
    else:
        width, height, xoffset, yoffset = 400, 300, 0, 0
        root.geometry("%dx%d%+d%+d" % (width, height, xoffset, yoffset))

    try:  
        app = Slideshow(root, image_filenames, slideshow_delay=2)
    except StopIteration:
        sys.exit("no image files found in %r" % (imagedir, ))

    root.bind("<Escape>", lambda _: root.destroy())  # exit on Esc
    root.bind('<Prior>', app.prev_image)
    root.bind('<Up>', app.prev_image)
    root.bind('<Left>', app.prev_image)
    root.bind('<Next>', app.next_image)
    root.bind('<Down>', app.next_image)
    root.bind('<Right>', app.next_image)

    root.bind("<Configure>", app.fit_image)  
    root.focus_set()
    root.mainloop()


if __name__ == '__main__':
    main()

