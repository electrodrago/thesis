from basicsr.utils import get_root_logger
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
import logging
from .video_recurrent_model import VideoRecurrentModel


root_path = "train_BasicVSR_REDS.yml"
opt, args = parse_options(root_path, is_train=True)
log_file = "a.txt"

logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)


model = VideoRecurrentModel(opt=opt)