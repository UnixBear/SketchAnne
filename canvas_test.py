#!/usr/bin/python

from tkinter import *

canvas_width = 600
canvas_height = 400

def paint( event ):
   python_green = "#476042"
   x1, y1 = ( event.x - 0 ), ( event.y - 0 )
   x2, y2 = ( event.x + 1 ), ( event.y + 1 )
   w.create_line(x1, y1, x2, y2, dash=(4,2))
   #w.create_oval( x1, y1, x2, y2, fill = python_green )
def line(event):
    w.create_line(300, 200, 500, 600)

master = Tk()
master.title( "Painting using Ovals" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
#w.bind( "<B1-Motion>", paint )
w.bind("<Button-1>", line)

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )
    
mainloop()
	