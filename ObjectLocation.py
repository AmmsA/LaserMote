__author__ = "Mustafa S"


# from Tkinter import *


class ObjectLocation(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.name = None
        self.description = None

    def get_first_point(self):
        return self.x1, self.y1

    def get_second_point(self):
        return self.x2, self.y2
        #
        # def ask_for_text(self):
        #
        #     master = Tk()
        #
        #     def show_entry_fields():
        #         print("Name: %s\nDescription: %s" % (e1.get(), e2.get()))
        #         master.destroy()
        #         #master.quit()
        #         self.name = e1.get()
        #         self.description = e2.get()
        #
        #
        #
        #     Label(master, text="Name").grid(row=0)
        #     Label(master, text="Description").grid(row=1)
        #
        #     e1 = Entry(master)
        #     e2 = Entry(master)
        #
        #     e1.grid(row=0, column=1)
        #     e2.grid(row=1, column=1)
        #
        #     Button(master, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=4)
        #
        #     mainloop()
