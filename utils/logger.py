from datetime import datetime

class Logger:
    def __init__(self, datetime_fmt="%Y-%m-%d %H:%M:%S"):
        self.format = datetime_fmt

    def log(self, text):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':', text)

    def end(self, text):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':', text)
        print()
