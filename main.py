from DCGAN import Generator
# from LAPGAN import LAPGenerator
from CGAN import CGenerator


def qtapp(models):
    from qtapp import Panorama
    from PySide2.QtWidgets import QApplication
    app = QApplication([])
    panorama = Panorama(models)
    app.exec_()

def main():
    CGAN = 'CGAN.pt'
    LAPGAN = ''
    DCGAN = 'DCGAN.pt'
    models = [CGAN, LAPGAN, DCGAN]
    qtapp(models)

if __name__ == '__main__':
    main()
