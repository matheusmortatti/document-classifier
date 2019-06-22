from tkinter import *
from tkinter import filedialog as fd
import threading
from pdf2image import convert_from_path
from skimage import transform
from skimage.io import imread
import numpy as np
import os
import keras
from keras.models import model_from_json
from shutil import copyfile

CS_NONE = 0
CS_LOADING_MODEL = 1
CS_CLASSIFYING = 2
CS_DONE = 3
CS_ERROR = 4

classes = ['balancete', 'balanco', 'contas', 'dre', 'luz A', 'luz B', 'nota fiscal']
  
class Application:
    def __init__(self, master=None):
        self.filepath = ""
        self.folderpath = ""
        self.classify_state = CS_NONE
        self.classify_thread = None
        self.model = None

        self.font = ("Arial", "10")
        self.firstcontainer = Frame(master)
        self.firstcontainer["pady"] = 10
        self.firstcontainer.pack()
  
        self.secondcontainer = Frame(master)
        self.secondcontainer["padx"] = 20
        self.secondcontainer.pack()
  
        self.thirdcontainer = Frame(master)
        self.thirdcontainer["padx"] = 20
        self.thirdcontainer.pack()
  
        self.fourthcontainer = Frame(master)
        self.fourthcontainer["pady"] = 20
        self.fourthcontainer.pack()

        #
        # Get Document.
        #
  
        self.title = Label(self.firstcontainer, text="Choose Document To Classify")
        self.title["font"] = ("Arial", "10", "bold")
        self.title.pack()

        self.choose_file = Button(self.firstcontainer)
        self.choose_file["text"] = "File"
        self.choose_file["font"] = ("Calibri", "8")
        self.choose_file["width"] = 12
        self.choose_file["command"] = self.GetFile
        self.choose_file.pack()
  
        self.filepath_message = Label(self.firstcontainer, text="", font=self.font)
        self.filepath_message.pack()

        #
        # Destination Folder
        #

        self.getfoldertitle = Label(self.secondcontainer, text="Choose Destination Folder")
        self.getfoldertitle["font"] = ("Arial", "10", "bold")
        self.getfoldertitle.pack()

        self.choose_folder = Button(self.secondcontainer)
        self.choose_folder["text"] = "Folder"
        self.choose_folder["font"] = ("Calibri", "8")
        self.choose_folder["width"] = 12
        self.choose_folder["command"] = self.GetFolder
        self.choose_folder.pack()
  
        self.folderpath_message = Label(self.secondcontainer, text="", font=self.font)
        self.folderpath_message.pack()

        #
        # Classify
        #

        self.classifytitle = Label(self.thirdcontainer, text="Classify!")
        self.classifytitle["font"] = ("Arial", "10", "bold")
        self.classifytitle.pack()

        self.classify_button = Button(self.thirdcontainer)
        self.classify_button["text"] = "Folder"
        self.classify_button["font"] = ("Calibri", "8")
        self.classify_button["width"] = 12
        self.classify_button["command"] = self.Classify
        self.classify_button.pack()
  
        self.classify_result_message = Label(self.thirdcontainer, text="", font=self.font)
        self.classify_result_message.pack()

        #
        # Error
        #
  
        self.error_message = Label(self.fourthcontainer, text="", font=self.font)
        self.error_message.pack()

    def Classify(self):
        self.error_message['text'] = ''
        self.error_message['fg'] = 'black'
        self.classify_result_message['text'] = ''
        self.classify_result_message['fg'] = 'black'

        if not self.VerifyFile():
            self.error_message['text'] = 'File chosen isn\'t a pdf or jpg file.'
            self.error_message['fg'] = 'red'
            return
        elif not self.VerifyFolder():
            self.error_message['text'] = 'Folder chosen is not valid'
            self.error_message['fg'] = 'red'
            return
        if self.classify_thread and self.classify_thread.isAlive():
            self.error_message['text'] = 'There\'s still a file being classified!'
            self.error_message['fg'] = 'red'
            return
        self.classify_thread = threading.Thread(target=self.LoadModelAndClassify)
        self.classify_thread.start()

    def GetFolder(self):
        self.error_message['text'] = ''
        self.error_message['fg'] = 'black'

        self.folderpath = fd.askdirectory()
        self.folderpath_message['text'] = self.folderpath
    
    def GetFile(self):
        self.error_message['text'] = ''
        self.error_message['fg'] = 'black'
        
        self.filepath = fd.askopenfilename()
        self.filepath_message['text'] = self.filepath

    def VerifyFile(self):
        if (not self.filepath) or not (self.filepath.endswith('pdf') or self.filepath.endswith('jpg')):
            return False
        return True
    def VerifyFolder(self):
        if not self.folderpath or self.folderpath == '':
            return False
        return True
    
    def LoadModelAndClassify(self):
        self.classify_state = CS_LOADING_MODEL
        self.classify_result_message['text'] = 'Loading Model ...'
        if not self.model:
            try:
                json_file = open(os.path.join(os.getcwd(),"model.json"), 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.model = model_from_json(loaded_model_json)
                # load weights into new model
                self.model.load_weights(os.path.join(os.getcwd(),"model.h5"))
            except:
                print(os.path.join(os.getcwd(),"model.h5"))
                self.classify_state = CS_ERROR
                self.error_message['text'] = 'Failed to load model.'
                self.error_message['fg'] = 'red'
                self.model = None
                return False
        
        self.classify_state = CS_CLASSIFYING
        self.classify_result_message['text'] = 'Classifying Document ...'

        images = []
        if self.filepath.endswith('jpg'):
                img = imread(self.filepath)
                img = self.preprocess_image(img)
                images.append(img)
        else:
                pages = self.SeparatePages(self.filepath)
                print(pages)
                i = 0
                for page in pages:
                    p = os.path.join(self.folderpath, str(i)) + '.jpg'
                    img = imread(p)
                    img = self.preprocess_image(img)
                    images.append(img)
                    os.remove(p)
                    i += 1
        images = np.asarray(images)
        print(images.shape)
        y_pred = self.model.predict(images)

        result = np.argmax(np.sum(y_pred))

        # Save document to path given + label name.
        doc_path = os.path.join(self.folderpath, classes[result])
        try:
            os.makedirs(doc_path)
        except FileExistsError:
            pass

        copyfile(self.filepath, os.path.join(doc_path, os.path.basename(self.filepath)))

        self.classify_result_message['text'] = 'Done! The model classified the document as \'' + classes[result] + '\''
        self.classify_result_message['fg'] = 'green'


    def SeparatePages(self, documentpath):
        print(documentpath)
        try:
            pages = convert_from_path(documentpath)
        except:
            self.error_message['text'] = 'Failed to process document.'
            self.error_message['fg'] = 'red'
            return None

        i = 0
        for page in pages:
            page.save(os.path.join(self.folderpath, str(i)) + '.jpg', 'JPEG')
            i += 1
        return pages

    def preprocess_image(self, frame):
        # Normalize Pixel Values
        normalized_frame = frame/255.0 - 0.5
        
        # Resize
        preprocessed_frame = transform.resize(normalized_frame, [150,150])
        
        # Create a 3-Channel image
    #     final_image = np.dstack((preprocessed_frame, preprocessed_frame, preprocessed_frame))
        
        return preprocessed_frame
        
  
root = Tk()
Application(root)
root.mainloop()
