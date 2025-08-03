#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Dataset Generator application.
"""

import sys
from PyQt5.QtWidgets import QApplication
from dataset_generator.gui.main_window import MainWindow

def main():
    """
    Main function to initialize and run the application.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
