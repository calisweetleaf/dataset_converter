from typing import List, Dict, Any, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QHelpEvent, QToolTip
from PyQt5.QtWidgets import (
    QTableWidget, QComboBox, QDialog, QVBoxLayout, QLabel,
    QScrollArea, QWidget, QPushButton, QHBoxLayout, QGroupBox,
    QFormLayout, QLineEdit
)

class DragDropTableWidget(QTableWidget):
    """TableWidget with drag and drop support for files."""
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data and mime_data.hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        mime_data = event.mimeData()
        if mime_data and mime_data.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data and mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.file_dropped.emit(file_path)

class CustomComboBox(QComboBox):
    """ComboBox with descriptions in tooltips."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.descriptions = {}

    def addItemWithDescription(self, text: str, description: str):
        self.addItem(text)
        self.descriptions[text] = description

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            help_event = QHelpEvent(event)
            view = self.view()
            if view:
                index = view.indexAt(help_event.pos())
                if index.isValid():
                    text = self.itemText(index.row())
                    if text in self.descriptions:
                        QToolTip.showText(help_event.globalPos(), self.descriptions[text])
                        return True
        return super().event(event)

class ProcessingWorker(QThread):
    """Worker thread for executing background processing tasks."""
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.task = None
        self.task_args = ()
        self.task_kwargs = {}

    def set_task(self, task, *args, **kwargs):
        self.task = task
        self.task_args = args
        self.task_kwargs = kwargs

    def run(self):
        try:
            if self.task:
                result = self.task(*self.task_args, **self.task_kwargs)
                self.finished_signal.emit(result)
            else:
                self.error_signal.emit("No task set for worker.")
        except Exception as e:
            self.error_signal.emit(str(e))

class CustomValidationRuleDialog(QDialog):
    """Dialog for defining custom validation rules."""
    def __init__(self, current_rules: List[Dict], all_fields: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Validation Rules")
        self.setGeometry(200, 200, 600, 400)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.all_fields = all_fields
        self.rules = current_rules
        self.rule_widgets = []

        self._init_ui()
        self._load_existing_rules()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        description_label = QLabel("Define custom rules for validating your dataset.")
        main_layout.addWidget(description_label)

        self.rules_scroll_area = QScrollArea()
        self.rules_scroll_area.setWidgetResizable(True)
        self.rules_container = QWidget()
        self.rules_layout = QVBoxLayout(self.rules_container)
        self.rules_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rules_scroll_area.setWidget(self.rules_container)
        main_layout.addWidget(self.rules_scroll_area)

        add_rule_button = QPushButton("Add New Rule")
        add_rule_button.clicked.connect(lambda: self._add_rule_widget())
        main_layout.addWidget(add_rule_button)

        button_box = QHBoxLayout()
        ok_button = QPushButton("Save Rules")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        main_layout.addLayout(button_box)

    def _add_rule_widget(self, rule_data: Optional[Dict] = None):
        rule_group = QGroupBox("Rule")
        rule_layout = QFormLayout(rule_group)

        field_combo = QComboBox()
        field_combo.addItem("ALL_FIELDS")
        field_combo.addItems(sorted(self.all_fields))
        rule_layout.addRow("Field:", field_combo)

        rule_type_combo = QComboBox()
        rule_type_combo.addItems(["Required", "Type Check", "Regex Match", "Min Length", "Max Length"])
        rule_layout.addRow("Rule:", rule_type_combo)

        value_input = QLineEdit()
        rule_layout.addRow("Value:", value_input)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self._remove_rule_widget(rule_group))
        rule_layout.addRow(remove_button)

        self.rules_layout.addWidget(rule_group)
        widget_set = {
            "group": rule_group, "field_combo": field_combo,
            "rule_type_combo": rule_type_combo, "value_input": value_input
        }
        self.rule_widgets.append(widget_set)

        if rule_data:
            field_combo.setCurrentText(rule_data.get("field", "ALL_FIELDS"))
            rule_type_combo.setCurrentText(rule_data.get("type", "Required"))
            value_input.setText(str(rule_data.get("value", "")))

    def _remove_rule_widget(self, group_to_remove: QGroupBox):
        group_to_remove.deleteLater()
        self.rule_widgets = [w for w in self.rule_widgets if w["group"] != group_to_remove]

    def _load_existing_rules(self):
        for rule in self.rules:
            self._add_rule_widget(rule)

    def get_rules(self) -> List[Dict]:
        collected_rules = []
        for widget_set in self.rule_widgets:
            if not widget_set["group"].isVisible(): continue
            rule = {
                "field": widget_set["field_combo"].currentText(),
                "type": widget_set["rule_type_combo"].currentText(),
                "value": widget_set["value_input"].text()
            }
            collected_rules.append(rule)
        self.rules = collected_rules
        return self.rules
