
SOFTMAX_OR_SIGMOIDS = 'sigmoids'  #multilabels softmax sigmoids

USE_DIRTY_LENS = False
USE_OCULAR_SURFACE = False
IMAGE_GRADABLE = True
IMAGE_LEFT_RIGHT = True
IMG_POSITION = False

LESION_SEG = False

DR1 = False

PORT_BASE = 20000

PORT_BASE_MULTI_LABEL = 10000

PORT_DEEP_SHAP = 25000

NUM_CLASSES = 30
LIST_THRESHOLD = [0.5 for _ in range(NUM_CLASSES)]

POSTPROCESS_EXCLUSION = True
POSTPROCESS_ALLNEGATIVE = True
POSTPROCESS_MULTI_POSITIVE = True

THRETHOLD_ALLNEGATIVE_CLASS0 = 0.45

ADD_BLACK_PIXEL_RATIO = 0.02

# PACS 文件存放位置
# BASE_DIR_PACS_SERVICE_INPUT = '/media/ubuntu/data2/ftp_fundus_image/upload'
# BASE_DIR_PACS_SERVICE_OUTPUT = '/media/ubuntu/data2/ftp_fundus_image/results'

BASE_DIR_PACS_SERVICE_INPUT = '/home/aaronkilik/dlp_pacs/upload'
BASE_DIR_PACS_SERVICE_OUTPUT = '/home/aaronkilik/dlp_pacs/results'

# BASE_DIR_PACS_SERVICE_INPUT = 'smb://10.12.193.3/ai/upload'
# BASE_DIR_PACS_SERVICE_OUTPUT = 'smb://10.12.193.3/ai/results'