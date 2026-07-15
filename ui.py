import sys
from main import pipeline
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QMainWindow, QPushButton,QDialog, QLineEdit,QLabel,QWidget,QVBoxLayout,QHBoxLayout,QComboBox
from models import Ranges,NoSurvivorsError
from PyQt6.QtCore import QThread, pyqtSignal
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
        self.strategy.addItems(["SMA","Order Block","Horizontal Volume"])
        self.ai=QComboBox()
        self.ai.addItems(["Volatility","Direction"])
        layout.addWidget(self.strategy)
        layout.addWidget(self.ai)
        self.setLayout(layout)
        self.strategy.currentIndexChanged.connect(self.submit_window.change_the_filling)
class SubmitWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.button = QPushButton("SUBMIT")
        self.button.clicked.connect(self.submit_the_data)
        layout = QHBoxLayout()
        
        self.sma_window=SMAWindow()
        self.order_block_window=OrderBlockWindow()
        self.order_block_window.hide()
        self.active_form=self.sma_window
        layout.addWidget(self.sma_window)
        layout.addWidget(self.order_block_window)
        layout.addWidget(self.button)
        self.setLayout(layout)
    def submit_the_data(self):
        
        self.active_form.submit(self.button)
        
    
        
    

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
            
        self.bar_label = QLabel("BAR=")
        self.bar_range = QLineEdit()
        self.bar_range.setPlaceholderText("e.g. 20,40 or 20,40,5")
        
        self.n_label = QLabel("N=")
        self.n_range = QLineEdit()
        self.n_range.setPlaceholderText("e.g. 20,40 or 20,40,5")
        
        self.leverage_label = QLabel("LEVERAGE=")
        self.leverage_range = QLineEdit()
        self.leverage_range.setPlaceholderText("e.g. 20,40 or 20,40,5")
        
        self.stop_label = QLabel("STOP=")
        self.stop_range = QLineEdit()
        self.stop_range.setPlaceholderText("e.g. 20,40 or 20,40,5(0.5%=5 in this)")
        
        self.take_label = QLabel("TAKE=")
        self.take_range = QLineEdit()
        self.take_range.setPlaceholderText("e.g. 20,40 or 20,40,5 (0.5%=5 in this)")
        
        
        
        self.long_window_label = QLabel("LONG WINDOW=")
        self.long_window_range = QLineEdit()
        self.long_window_range.setPlaceholderText("e.g. 20,40 or 20,40,5")   
        layout_sma=QVBoxLayout()

        layout_bar=QHBoxLayout()
        layout_bar.addWidget(self.bar_label)
        layout_bar.addWidget(self.bar_range)

        layout_n=QHBoxLayout()
        layout_n.addWidget(self.n_label)
        layout_n.addWidget(self.n_range)

        layout_long_window=QHBoxLayout()
        layout_long_window.addWidget(self.long_window_label)
        layout_long_window.addWidget(self.long_window_range)

        layout_take=QHBoxLayout()
        layout_take.addWidget(self.take_label)
        layout_take.addWidget(self.take_range)

        layout_stop=QHBoxLayout()
        layout_stop.addWidget(self.stop_label)
        layout_stop.addWidget(self.stop_range)

        layout_leverage=QHBoxLayout()
        layout_leverage.addWidget(self.leverage_label)
        layout_leverage.addWidget(self.leverage_range)
        

        layout_sma.addLayout(layout_bar)
        layout_sma.addLayout(layout_n)
        layout_sma.addLayout(layout_long_window)
        layout_sma.addLayout(layout_take)
        layout_sma.addLayout(layout_stop)
        layout_sma.addLayout(layout_leverage)
        
        self.fields = {
            "BAR": self.bar_range,
            "N": self.n_range,
            "LONG_WINDOW": self.long_window_range,
            "TAKE": self.take_range,
            "STOP": self.stop_range,
            "LEVERAGE": self.leverage_range
            }
        

        self.setLayout(layout_sma)
    def get_values(self,edit_line):
        split_text=edit_line.text().split(",")

        try:
            if len(split_text)==3:
                range_var=range(int(split_text[0]),int(split_text[1]),int(split_text[2]))
                return range_var
            if len(split_text)==2:
                range_var=range(int(split_text[0]),int(split_text[1]))
                return range_var
            else:
                return None
            
        except ValueError:
            
            
            
            
            return None
        except IndexError:
           
           
            return None
        
        
        
    def submit(self,button):
        self.submit_button = button
        results={}
        errors=[]
        for  name, edit_line in self.fields.items():
            result=self.get_values(edit_line)
            if result is None:
                errors.append(name)
            else:
                results[name]=result
            

            
        if errors :
            QMessageBox.warning(self, "Invalid input",
                f"Invalid input in: {', '.join(errors)}\nExpected e.g. 20,40 or 20,40,5")
            return
        else:
            self.pipeline_thread = PipelineThread(results)
            self.pipeline_thread.finished_ok.connect(self.on_pipeline_done)
            self.pipeline_thread.failed.connect(self.on_pipeline_failed)
            self.submit_button.setEnabled(False)
            self.pipeline_thread.start()
            print(results)
    def on_pipeline_done(self, best_params):
        self.submit_button.setEnabled(True)
        print("ALL WENT WELL")

    def on_pipeline_failed(self, message):
        self.submit_button.setEnabled(True)
        QMessageBox.warning(self, "No survivors",
                f" {message}")
            
    
class OrderBlockWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WAIT")
        self.text=QLabel("Under construction...")
        self.ob_layout=QHBoxLayout()
        self.ob_layout.addWidget(self.text)
        self.setLayout(self.ob_layout)
    def get_values(self):
            
            adx=67
            return adx
    def submit(self,button):
        adx=self.get_values()
        print(adx)


class PipelineThread(QThread):
    finished_ok = pyqtSignal(object)   # custom signals: CLASS attributes,
    failed      = pyqtSignal(str)      # typed by payload

    def __init__(self, ranges):
        super().__init__()
        self.results = ranges

    def run(self):
        try:
            best_params=pipeline(Ranges(leverage=self.results["LEVERAGE"],stop=self.results["STOP"],take=self.results["TAKE"],n=self.results["N"],bar=self.results["BAR"],long_window=self.results["LONG_WINDOW"]))
            
            self.finished_ok.emit(best_params)

            
        except NoSurvivorsError as e:
            
            self.failed.emit( str(e)) 
            

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()