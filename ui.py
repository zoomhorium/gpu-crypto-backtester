import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QMainWindow, QPushButton,QDialog, QLineEdit,QLabel,QWidget,QVBoxLayout,QHBoxLayout,QComboBox


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.full_window = FullWindow()
        self.setCentralWidget(self.full_window)
        
class FullWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.submit_window = SubmitWindow()
        self.choise_window = ChoiseWindow(self.submit_window)
        
        layout.addWidget(self.choise_window)
        layout.addWidget(self.submit_window)
        self.setLayout(layout)
class ChoiseWindow(QWidget):
    def __init__(self, submit_window):
        
        super().__init__()
        self.submit_window = submit_window
        layout = QVBoxLayout()
        self.strategy=QComboBox()
        self.strategy.addItems(["SMA","Order Block","Horizonal Volume"])
        self.ai=QComboBox()
        self.ai.addItems(["Volatility","Derection"])
        layout.addWidget(self.strategy)
        layout.addWidget(self.ai)
        self.setLayout(layout)
        self.strategy.currentIndexChanged.connect(self.submit_window.change_the_filling)
class SubmitWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        button = QPushButton("SUBMIT")
        button.clicked.connect(self.submit_the_data)
        layout = QHBoxLayout()
        
        self.sma_window=SMAWindow()
        self.order_block_window=OrderBlockWindow()
        self.order_block_window.hide()
        self.active_form=self.sma_window
        layout.addWidget(self.sma_window)
        layout.addWidget(self.order_block_window)
        layout.addWidget(button)
        self.setLayout(layout)
    def submit_the_data(self):
        self.active_form.submit()
        
    
        
    

    def change_the_filling(self,i):
        
        if (i==0):
            self.order_block_window.hide()
            self.sma_window.show()
            self.active_form=self.sma_window
        else:
            self.order_block_window.show()
            self.active_form=self.order_block_window
            self.sma_window.hide()
    
class SMAWindow (QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMA")
            
        self.adx_lable = QLabel("ADX=")
        self.adx_range = QLineEdit()
        self.adx_range.setPlaceholderText("ENTER ADX RANGE WITH COMA IN BETWEAN")
            
        layout_adx=QHBoxLayout()
        layout_adx.addWidget(self.adx_lable)
        layout_adx.addWidget(self.adx_range)
        self.setLayout(layout_adx)
    def get_values(self):
        splited_text=self.adx_range.text().split(",")
        try:
            adx=range(int(splited_text[0]),int(splited_text[1]))
            return adx
        except ValueError:
            
            dlg =QMessageBox.warning(self, "Invalid input", "Expected e.g. 20,40")
            
            
            return None
        except IndexError:
            dlg =QMessageBox.warning(self, "Invalid input", "Expected e.g. 20,40")
           
            return None
        
        
        
    def submit(self):
        adx=self.get_values()
        if adx is None:
            return
        print(adx)
    
class OrderBlockWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WAIT")
        self.text=QLabel("Under constraction...")
        self.ob_layout=QHBoxLayout()
        self.ob_layout.addWidget(self.text)
        self.setLayout(self.ob_layout)
    def get_values(self):
            
            adx=67
            return adx
    def submit(self):
        adx=self.get_values()
        print(adx)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()