import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from functools import partial
from tkinter.filedialog import askopenfile, askopenfilename
from keras.models import load_model, Model
import cv2
import numpy as np
from keras import backend as K
import time
from PIL import Image, ImageTk

class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        labelframe1 = LabelFrame(self, bg="lightcyan",bd=30,height=100)
        labelframe1.pack(fill='both',expand=True)

        # Membuat StringVar() untuk menampilkan teks 'Sistem Deteksi Tumor'
        self.dialog_var = StringVar()
        self.dialog_var.set("Sistem Deteksi Tumor")

        toplabel = Label(labelframe1, font=("Courier", 15), height=2, textvariable=self.dialog_var, fg="red", bg="lightcyan")
        toplabel.pack()

        button1 = Button(labelframe1, text="Mulai", command=lambda: controller.show_page(PageTwo),width=10, height=1,bg="green", fg="white")
        button1.pack()
        button_exit = Button(labelframe1, text="Exit", command=self.exit_application, width=10, height=1,bg="red", fg="white")
        button_exit.pack()

        # Menempatkan label, tombol Mulai, dan tombol Exit di tengah halaman
        labelframe1.place(relx=0.5, rely=0.5, anchor='center')

    def exit_application(self):
        self.controller.quit()
        
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.dialog_var = StringVar()
        self.dialog_var.set("Selamat Datang")
        
        mainframe = Frame(self,bd=20)
        mainframe.pack()

        topframe = Frame(mainframe, bd=10)
        topframe.pack()
        middle_frame = Frame(mainframe, bd=10)
        middle_frame.pack()
        bottom_frame = Frame(mainframe, bd=10)
        bottom_frame.pack()
        notification_frame = Frame(mainframe, bd=10)
        notification_frame.pack()

        btn_h5_fopen = Button(topframe, text='Input Model', command=lambda: self.open_file(controller), bg="black", fg="white")
        btn_h5_fopen.grid(row=2, column=1)

        self.h5_var = StringVar()
        self.h5_var.set("/")
        h5_entry = Entry(topframe, textvariable=self.h5_var, width=40)
        h5_entry.grid(row=2, column=2)

        btn_h5_confirm = Button(topframe, text='Load Model', command=self.load_weights, bg="black", fg="white")
        btn_h5_confirm.grid(row=2, column=4)

        btn_img_fopen = Button(topframe, text='Input Gambar', command=lambda: self.open_image(controller), bg="black", fg="white")
        btn_img_fopen.grid(row=7, column=1)

        self.img_var = StringVar()
        self.img_var.set("/")
        img_entry = Entry(topframe, textvariable=self.img_var, width=40)
        img_entry.grid(row=7, column=2)

        btn_img_confirm = Button(topframe, text='Load Gambar', command=self.load_image, bg="black", fg="white")
        btn_img_confirm.grid(row=7, column=4)

        ml = Label(middle_frame, font=("Courier", 10), bg="gray", fg="white", text="Gambar Akan Tampil Di Bawah Ini").grid(row=1, column=1)

        self.img_label = Label(middle_frame, padx=10, pady=10)
        self.img_label.grid(row=3, column=1)

        btn_test = Button(bottom_frame, text='Deteksi Jenis Tumor', command=self.test_image, bg="green", fg="white")
        btn_test.pack()
        
        self.test_result_var = StringVar()
        self.test_result_var.set("Hasil Deteksi Akan Tampil Di Sini")
        test_result_label = Label(bottom_frame, font=("Courier", 20), height=3, textvariable=self.test_result_var, bg="white", fg="purple").pack()
        labelframe1 = LabelFrame(notification_frame, text="Notification Box", bg="yellow")
        labelframe1.pack()

        toplabel = Label(labelframe1, font=("Courier", 15), height=2, textvariable=self.dialog_var, fg="red", bg="lightcyan")
        toplabel.pack()
        buttonbmenu = Button(notification_frame, text="Menu", command=lambda: controller.show_page(PageOne),bg="grey", fg="white",width=10, height=1)
        buttonbmenu.pack()
        mainframe.place(relx=0.5, rely=0.5, anchor='center')
    def open_file(self, controller):
        file_path = askopenfilename(initialdir='/', filetypes=[('Model Weights', '*.h5')])
        self.dialog_var.set("Model Di Input,Tapi Belum Di Load")
        self.h5_var.set(file_path)

    def load_weights(self):
        self.dialog_var.set("Model Di Input.......")
        weight_path = self.h5_var.get()
        global model, height, width, channel
        model = load_model(weight_path)
        model.summary()
        load_input = model.input
        input_shape= list(load_input.shape)
        height = int(input_shape[1])
        width = int(input_shape[2])
        channel = int(input_shape[3])
        print(height, width, channel)
        self.dialog_var.set("Model Telah Di Load!")
        return
    def open_image(self, controller):
        file_path = askopenfilename(initialdir='/', filetypes=[('Image File', '*.*')])
        self.dialog_var.set("Gambar Telah Di input,tapi belum di Load")
        self.img_var.set(file_path)

        image = Image.open(file_path)
        image = image.resize((256, 256))
        photo = ImageTk.PhotoImage(image)
        self.img_label.configure(image=photo)
        self.img_label.image = photo

    def load_image(self):
        self.dialog_var.set("Image loading.............")
        path = self.img_var.get()
        global imgs
        if channel == 1:
            imgs = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            imgs = cv2.imread(path)
        imgs = cv2.resize(imgs, (150, 150))  # resize image to 150x150
        imgs = imgs.reshape(1, 150, 150, channel).astype('float32')/ 255.0
        imgs = np.array(imgs) / 255
        print(imgs.shape)
        self.dialog_var.set("Gambar Telah Di input!")

    def test_image(self):
        self.dialog_var.set("Deteksi Gambar.............")
        result_text = "Jenis Tumor Pada Gambar: "
        img_path = self.img_var.get()
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        class_names = ['glioma', 'meningioma', 'normal', 'pituitary']
        mapped_class_name = class_names[predicted_class_index]
    
        result_text += mapped_class_name
        self.test_result_var.set(result_text)
        self.dialog_var.set("Gambar Berhasil Di Deteksi")

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Sistem Deteksi Tumor")
        self.geometry("720x720")
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.pages = {}

        for Page in (PageOne, PageTwo):
            page = Page(self.container, self)
            self.pages[Page] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page(PageOne)

    def show_page(self, cont):
        page = self.pages[cont]
        page.tkraise()

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
    print("finished")