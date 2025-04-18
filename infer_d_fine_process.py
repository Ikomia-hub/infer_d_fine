import copy
from ikomia import core, dataprocess, utils
from infer_d_fine.utils.model_loading import load_model
from infer_d_fine.utils.class_lists import obj365_classes, coco_classes
import torch
from PIL import Image
import torchvision.transforms as T


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDFineParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "dfine_m"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.5
        self.pretrained_dataset = "obj2coco"
        self.update = False
        self.model_weight_file = ""
        self.config_file = ""
        self.class_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.pretrained_dataset = str(param_map["pretrained_dataset"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.config_file = str(param_map["config_file"])
        self.class_file = str(param_map["class_file"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["model_weight_file"] = str(self.model_weight_file)
        param_map["pretrained_dataset"] = str(self.pretrained_dataset)
        param_map["config_file"] = str(self.config_file)
        param_map["class_file"] = str(self.class_file)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDFine(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Create parameters object
        if param is None:
            self.set_param_object(InferDFineParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.model = None
        self.postprocessor = None
        self.labels = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def set_class_names(self, param):
        if param.model_weight_file:
            if not param.config_file:
                raise ValueError(
                    "The 'config_file' is required when using a custom model file.")
            else:
                # load class names from file .txt
                with open(param.class_file, 'r') as f:
                    self.labels = [line.strip() for line in f]
        else:
            if param.pretrained_dataset == 'obj365':
                self.labels = obj365_classes
            else:
                # Objects365+COCO means finetuned model on COCO using pretrained weights trained on Objects365.
                self.labels = coco_classes
        # Set class names
        self.set_names(self.labels)

    def infer(self, img_array, size):

        # Convert numpy array to PIL Image
        im_pil = Image.fromarray(img_array).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        # Define transformations
        transforms = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

        # Apply transformations and prepare the input tensor
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        # Perform inference
        outputs = self.model(im_data)
        outputs = self.postprocessor(outputs, orig_size)
        labels, boxes, scores = outputs

        return labels, boxes, scores

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")

            self.model, self.postprocessor = load_model(param, self.device)

            # Set classe names
            self.set_class_names(param)
            param.update = False

        labels, boxes, scores = self.infer(src_image, param.input_size)

        # Process all objects exceeding the confidence threshold
        for label, box, conf in zip(labels, boxes, scores):
            conf = conf.detach().cpu().numpy()
            box = box.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            for j, scr in enumerate(conf):
                if scr > param.conf_thres:  # Compare confidence score with threshold
                    x1, y1, x2, y2 = box[j][0], box[j][1], box[j][2], box[j][3]
                    width = x2 - x1
                    height = y2 - y1
                    self.add_object(
                        j,  # Index of the current object
                        int(label[j]),
                        float(scr),
                        float(x1),
                        float(y1),
                        float(width),
                        float(height)
                    )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDFineFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_d_fine"
        self.info.short_description = "Inference with D-FINE models"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Y. Peng, H. Li, P. Wu, Y. Zhang, X. Sun and F. Wu"
        self.info.article = "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement"
        self.info.journal = "arXiv"
        self.info.year = 2024
        self.info.license = "Apache 2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.13.0"

        # Python compatibility
        self.info.min_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2410.13842"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_d_fine"
        self.info.original_repository = "https://github.com/Peterande/D-FINE"

        # Keywords used for search
        self.info.keywords = "DETR, object, detection, real-time"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferDFine(self.info.name, param)
