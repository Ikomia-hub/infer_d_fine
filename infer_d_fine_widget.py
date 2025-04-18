from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_d_fine.infer_d_fine_process import InferDFineParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available

# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------


class InferDFineWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDFineParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("dfine_n")
        self.combo_model.addItem("dfine_s")
        self.combo_model.addItem("dfine_m")
        self.combo_model.addItem("dfine_l")
        self.combo_model.addItem("dfine_x")

        self.combo_model.setCurrentText(self.parameters.model_name)

        # Pretained dataset name
        self.combo_dataset = pyqtutils.append_combo(
            self.grid_layout, "Pretrained dataset")
        self.combo_dataset.addItem("coco")
        self.combo_dataset.addItem("obj2coco")
        self.combo_dataset.addItem("obj365")

        self.combo_dataset.setCurrentText(self.parameters.pretrained_dataset)

        # Hyper-parameters for custom weight
        custom_weight = bool(self.parameters.model_weight_file)
        self.check_cfg = QCheckBox("Custom model")
        self.check_cfg.setChecked(custom_weight)
        self.grid_layout.addWidget(
            self.check_cfg, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_cfg.stateChanged.connect(self.on_custom_weight_changed)

        # Model weight file selection
        self.label_hyp = QLabel("Model weight (.pt)")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.model_weight_file,
            tooltip="Select model weight file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        # Config file selection
        self.label_config_file = QLabel("Config file (.yaml)")
        self.browse_config_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.config_file,
            tooltip="Select config file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_config_file, row, 0)
        self.grid_layout.addWidget(self.browse_config_file, row, 1)

        # Class file selection
        self.label_class_file = QLabel("Class file (.txt)")
        self.browse_class_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.class_file,
            tooltip="Select class file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_class_file, row, 0)
        self.grid_layout.addWidget(self.browse_class_file, row, 1)

        # Ensure visibility based on custom weight toggle
        self.label_hyp.setVisible(custom_weight)
        self.browse_weight_file.setVisible(custom_weight)
        self.label_config_file.setVisible(custom_weight)
        self.browse_config_file.setVisible(custom_weight)
        self.label_class_file.setVisible(custom_weight)
        self.browse_class_file.setVisible(custom_weight)

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        # Input size
        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout,
            "Input size",
            self.parameters.input_size
        )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_weight_changed(self, int):
        self.label_hyp.setVisible(self.check_cfg.isChecked())
        self.browse_weight_file.setVisible(self.check_cfg.isChecked())
        self.label_config_file.setVisible(self.check_cfg.isChecked())
        self.browse_config_file.setVisible(self.check_cfg.isChecked())
        self.label_class_file.setVisible(self.check_cfg.isChecked())
        self.browse_class_file.setVisible(self.check_cfg.isChecked())

    def on_apply(self):
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.combo_dataset = self.combo_dataset.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        if self.check_cfg.isChecked():
            self.parameters.model_weight_file = self.browse_weight_file.path
            self.parameters.config_file = self.browse_config_file.path
            self.parameters.class_file = self.browse_class_file.path
        self.parameters.update = True
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDFineWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_d_fine"

    def create(self, param):
        # Create widget object
        return InferDFineWidget(param, None)
