from PIL import Image
import torch
from torch.autograd import Variable
from torch import Tensor, FloatTensor
from torchvision.utils import save_image
import numpy as np
import re
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QLabel, QLineEdit, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy
from PySide2.QtGui import QImage, QPixmap
import pickle
class Panorama(QWidget):
    def __init__(self, models):
        super(Panorama, self).__init__()

        sp = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.setWindowTitle('Emoji generator')
        self.setGeometry(400, 400, 600, 600)
        self.layout = QVBoxLayout()
        self.bLayout = QHBoxLayout()
        self.bCGAN = QPushButton('CGAN')
        self.bLAPGAN = QPushButton('LAPGAN')
        self.bDCGAN = QPushButton('DCGAN')
        # self.bLayout.addWidget(self.bCGAN)
        # self.bLayout.addWidget(self.bLAPGAN)
        self.bLayout.addWidget(self.bDCGAN)
        self.imglabel = QLabel()
        self.imglabel.setAlignment(Qt.AlignCenter)
        self.imglayout = QHBoxLayout()
        self.imglayout.addWidget(self.imglabel)
        self.textinput = QLineEdit()
        self.cganlayout = QHBoxLayout()
        self.cganlayout.addWidget(self.textinput)
        self.cganlayout.addWidget(self.bCGAN)
        self.layout.addLayout(self.cganlayout)
        self.layout.addLayout(self.bLayout)
        self.layout.addLayout(self.imglayout)
        # self.layout.addWidget(self.imglabel)
        self.setLayout(self.layout)
        self.show()
        
        self.setup_callback()

        self.load_model(models)
        # self.plot_img()

    def setup_callback(self):
        self.bCGAN.clicked.connect(self.run_CGAN)
        self.bLAPGAN.clicked.connect(self.run_LAPGAN)
        self.bDCGAN.clicked.connect(self.run_DCGAN)

    def run_CGAN(self):
        self.img_shape = 64
        self.latent_dim = 100
        self.embedding_dim = 768
        z = Variable(FloatTensor(np.random.normal(0, 1, (self.img_shape, self.latent_dim))))
        # gen_embeddings = Variable(FloatTensor(np.random.normal(0, 1, (self.img_shape, self.embedding_dim))))
        # text = re.split(r' +', self.textinput.text().strip())
        text = self.textinput.text().strip()
        gen_embeddings = self.get_text_embeddings(text, "google_list")
        embeddings = gen_embeddings.unsqueeze_(0).repeat([z.size()[0], 1])

        # print(z.size(), embeddings.size())
        gen_imgs = self.CGAN(z, embeddings)
        save_image(gen_imgs.data[:25], "ganoutput.png", nrow=5, normalize=True)
        self.load_image()

    def run_LAPGAN(self):
        self.img_shape = 64
        self.latent_dim = 500
        z = Variable(Tensor(np.random.normal(0, 1, (self.img_shape, self.latent_dim))))
        gen_imgs = self.LAPGAN(z)
        # print(gen_imgs)
        self.load_image(gen_imgs.data[:25])

    def get_text_embeddings(text, src):
      if src == 'google':
        sentence_embedding = []
        tokens = text.replace("-", " ")
        tokens = tokens.split(' ')
        for token in tokens:
          if token.lower() not in WORD_VEC.vocab:
            #print(token, " is not in vocab")
            continue
          token_embedding = WORD_VEC[token.lower()]
          sentence_embedding.append(token_embedding)
        #print(np.array(sentence_embedding).shape)
        avg_embeddings = np.mean(np.array(sentence_embedding), 0)
      elif  src == 'bert':
        tokens = TOKENIZER.tokenize(text)
        indexed_tokens = TOKENIZER.convert_tokens_to_ids(tokens)
        if device.type == 'cuda':
          hidden_output = NLP_MODEL(torch.tensor([indexed_tokens]))
        else:  
          hidden_output = NLP_MODEL(torch.tensor([indexed_tokens]))
        embeddings = hidden_output[0][0]
        avg_embeddings = torch.mean(embeddings, dim = 0)
        return avg_embeddings
      elif src == 'google_list':
        avg_embeddings = GOOGLE_EMBEDDING[text]
      elif src == 'bert_list':
        avg_embeddings = BERT_EMBEDDING[text]
      return FloatTensor(avg_embeddings)

    
    def run_DCGAN(self):
        self.img_shape = 64
        self.latent_dim = 5
        z = Variable(Tensor(np.random.normal(0, 1, (self.img_shape, self.latent_dim))))
        gen_imgs = self.DCGAN(z)
        save_image(gen_imgs.data[:25], "ganoutput.png", nrow=5, normalize=True)
        self.load_image()

    def load_model(self, models):
        import random
        manualSeed = random.randint(1, 10000) # use if you want new results
        # print("Random Seed: ", manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        cgan, lapgan, dcgan = models
        # self.CGAN = torch.load(cgan).eval()
        # self.LAPGAN = torch.load(lapgan).eval()
        self.load_CGAN(cgan)
        self.load_DCGAN(dcgan)

    def load_CGAN(self, cgan):
        # from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer
        # self.TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.NLP_MODEL = BertModel.from_pretrained('bert-base-uncased',
        #                                       output_attentions=False,
        #                                       output_hidden_states=True)
        self.CGAN = torch.load(cgan, map_location='cpu').eval()

    def load_DCGAN(self, dcgan):
        self.DCGAN = torch.load(dcgan, map_location='cpu').eval()


    def load_image(self):
        self.img = Image.open("ganoutput.png")
        self.pix = np.asarray(self.img)
        self.cyl = self.pix
        self.plot_img()

    def plot_img(self):
        height, width, channel = self.cyl.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.cyl, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qImg)
        self.imglabel.setPixmap(self.pixmap)
