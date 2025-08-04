import json
import os
from typing import List, Dict, Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QTabWidget, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox, QSpinBox,
    QCheckBox, QAction, QToolBar, QDialog, QFormLayout, QCompleter, QStringListModel
)

from dataset_generator.core.data_loader import load_data
from dataset_generator.core.data_saver import save_data
from dataset_generator.core.data_processor import process_data
from dataset_generator.gui.components import (
    DragDropTableWidget, CustomComboBox, ProcessingWorker, CustomValidationRuleDialog
)
from dataset_generator.utils.constants import (
    logger, INPUT_FORMATS, OUTPUT_FORMATS, DATASET_TEMPLATES, FAKER_AVAILABLE
)

class MainWindow(QMainWindow):
    """Main application window for the Dataset Generator GUI."""

    def __init__(self):
        super().__init__()
        self.input_data: List[Dict[str, Any]] = []
        self.output_data: List[Dict[str, Any]] = []
        self.current_template: str = ""
        self.field_mapping: Dict[str, str] = {}
        self.current_operation: str = ""

        self.setWindowTitle("Dataset Generator for LLM Fine-Tuning")
        self.setGeometry(100, 100, 1200, 800)
        self._apply_theme()
        self.worker = ProcessingWorker()
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.error_signal.connect(self.show_error)
        self._create_central_widget()

    def _apply_theme(self):
        # ... (theme logic as before)
        pass

    def _create_central_widget(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_convert_tab(), "Convert")
        # Add other tabs later
        layout.addWidget(self.tabs)

    def _create_convert_tab(self):
        """Create the Convert tab for format conversion."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input
        input_group = QGroupBox("Input")
        input_layout = QFormLayout(input_group)
        self.input_path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_input_file)
        self.input_format_combo = CustomComboBox()
        for fmt, desc in INPUT_FORMATS.items(): self.input_format_combo.addItemWithDescription(fmt, desc)
        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self._load_dataset)
        input_layout.addRow("File:", self.input_path_edit)
        input_layout.addRow(browse_button)
        input_layout.addRow("Format:", self.input_format_combo)
        input_layout.addRow(load_button)
        layout.addWidget(input_group)

        # Process
        process_group = QGroupBox("Process")
        process_layout = QFormLayout(process_group)
        self.template_combo = CustomComboBox()
        for name, info in DATASET_TEMPLATES.items():
            self.template_combo.addItemWithDescription(name, info["description"])
        self.map_fields_button = QPushButton("Map Fields")
        self.map_fields_button.setEnabled(False)
        self.map_fields_button.clicked.connect(self._show_field_mapping_dialog)
        process_layout.addRow("Template:", self.template_combo)
        process_layout.addRow(self.map_fields_button)
        layout.addWidget(process_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QFormLayout(output_group)
        self.output_path_edit = QLineEdit()
        output_browse_button = QPushButton("Browse...")
        output_browse_button.clicked.connect(self._browse_output_file)
        self.output_format_combo = CustomComboBox()
        for fmt, desc in OUTPUT_FORMATS.items(): self.output_format_combo.addItemWithDescription(fmt, desc)
        self.convert_button = QPushButton("Convert & Save")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self._convert_dataset)
        output_layout.addRow("File:", self.output_path_edit)
        output_layout.addRow(output_browse_button)
        output_layout.addRow("Format:", self.output_format_combo)
        output_layout.addRow(self.convert_button)
        layout.addWidget(output_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = DragDropTableWidget()
        self.preview_table.file_dropped.connect(self._handle_dropped_file)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group)

        return tab

    def _browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File")
        if path:
            self.input_path_edit.setText(path)
            ext = os.path.splitext(path)[1].lower().replace('.', '')
            if ext in INPUT_FORMATS:
                self.input_format_combo.setCurrentText(ext)

    def _browse_output_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save File")
        if path:
            self.output_path_edit.setText(path)
            ext = os.path.splitext(path)[1].lower().replace('.', '')
            if ext in OUTPUT_FORMATS:
                self.output_format_combo.setCurrentText(ext)

    def _handle_dropped_file(self, path):
        self.input_path_edit.setText(path)
        self._load_dataset()

    def _load_dataset(self):
        path = self.input_path_edit.text()
        if not path:
            return self.show_error("Input file path is required.")
        self.current_operation = "load"
        self.worker.set_task(load_data, file_path=path, format_type=self.input_format_combo.currentText())
        self.worker.start()

    def _convert_dataset(self):
        # First, process the data with the mapping
        self.current_operation = "process"
        self.worker.set_task(process_data, input_data=self.input_data, template_name=self.current_template, mapping=self.field_mapping)
        self.worker.start()

    def _save_converted_data(self):
        path = self.output_path_edit.text()
        if not path:
            return self.show_error("Output file path is required.")
        self.current_operation = "save"
        self.worker.set_task(save_data, data=self.output_data, output_path=path, format_type=self.output_format_combo.currentText())
        self.worker.start()

    def _show_field_mapping_dialog(self):
        if not self.input_data:
            return self.show_error("Please load data first.")

        template_name = self.template_combo.currentText()
        if not template_name or template_name == "custom":
            return self.show_error("Please select a valid template (not custom).")

        # Get all unique fields from the loaded data
        all_input_fields = set()
        for record in self.input_data:
            all_input_fields.update(record.keys())

        dialog = QDialog(self)
        dialog.setWindowTitle("Map Fields")
        layout = QFormLayout(dialog)

        template_fields = list(DATASET_TEMPLATES[template_name]["structure"].keys())
        self.mapping_combos = {}

        for field in template_fields:
            combo = QComboBox()
            combo.addItems([""] + sorted(list(all_input_fields)))
            self.mapping_combos[field] = combo
            layout.addRow(f"'{field}':", combo)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addRow(ok_button)

        if dialog.exec_() == QDialog.DialogCode.Accepted:
            self.field_mapping = {}
            for field, combo in self.mapping_combos.items():
                if combo.currentText():
                    self.field_mapping[field] = combo.currentText()
            self.current_template = template_name
            self.convert_button.setEnabled(True)
            logger.info(f"Field mapping set: {self.field_mapping}")

    def _update_preview_table(self):
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        if not self.input_data:
            return

        headers = list(self.input_data[0].keys())
        self.preview_table.setColumnCount(len(headers))
        self.preview_table.setHorizontalHeaderLabels(headers)

        for i, row in enumerate(self.input_data[:100]): # Preview first 100
            self.preview_table.insertRow(i)
            for j, key in enumerate(headers):
                self.preview_table.setItem(i, j, QTableWidgetItem(str(row.get(key, ""))))
        self.preview_table.resizeColumnsToContents()

    def process_finished(self, result):
        logger.info(f"Worker finished operation: {self.current_operation}")
        if self.current_operation == "load":
            self.input_data = result
            self._update_preview_table()
            self.map_fields_button.setEnabled(True)
        elif self.current_operation == "process":
            self.output_data = result
            # Now save the processed data
            self._save_converted_data()
        elif self.current_operation == "save":
            QMessageBox.information(self, "Success", "Dataset saved successfully.")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        logger.error(message)
