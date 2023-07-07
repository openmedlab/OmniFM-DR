import os
import sys
import re
import cv2
import numpy
import torch
import gensim
from gensim import corpora
from loguru import logger
import numpy as np
from PIL import Image
from torchvision import transforms
from tasks.mm_tasks.omnifmdr import OmniFMDrTask
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level='INFO')
logger.add('./results/log/file_{time}.log', enqueue=True)

JOINT_CHECKPOINT='./checkpoints/omnifmdr.pt'

cfg=None
generator = None
bos_item = None
eos_item = None
pad_idx = None
task = None
models = None
use_cuda = torch.cuda.is_available()
use_fp16 = False
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
patch_resize_transform = None


def init_task():
    global generator
    global bos_item
    global eos_item
    global pad_idx
    global task
    global models
    global patch_resize_transform
    global cfg
    
    target_task = 'omnifm_dr'
    tasks.register_task(target_task, OmniFMDrTask)

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = [
        "",
        "--task={}".format(target_task),
        "--beam=5",
        "--path={}".format(JOINT_CHECKPOINT),
        "--bpe-dir=utils/BPE",
        "--no-repeat-ngram-size=3", 
        "--patch-image-size=512"
    ]

    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)

    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)

    models, cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task=task
    )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()


def get_examples():
    demo_path = './results/demo'
    images = [
        '19759491_51878257.jpg', 
        '15114531_50027153.jpg', 
        '17206933_51664027.jpg', 
        '11022245_58274962.jpg',
        '12530259_54170209.jpg', 
        '18309149_54224807.jpg', 
        '12145137_54833205.jpg'
    ]
    vg_examples = [
        ['Pleural Effusion', 'Atelectasis', 'Pneumothorax'],
        ['Pleural Effusion', 'Pneumonia', 'Pneumothorax'],
        ['Edema', 'Pleural Effusion', 'Pneumonia'],
        ['Pleural Effusion', 'Pneumonia', 'Pneumothorax'],
        ['Atelectasis', 'Pneumothorax', 'nodule'],
        ['Atelectasis',  'Pneumothorax', 'Pleural Effusion', 'nodule'],
        ['nodule', 'Pleural Effusion', 'Pneumothorax']
    ]
    radio_example = []
    for im, vgq in zip(images, vg_examples):
        img = os.path.join(demo_path, im)
        q = [f'where is {cla}?' for cla in vgq]
        q = "&&".join(q)
        q = 'what can we get from this chest medical image? && what disease does this image have?' + '&&' + q 
        radio_example.append([img, q])
        
    return radio_example


def image_preprocess(img, target_size=None):
    if target_size is not None:
        W, H = img.size
        img = img.resize(target_size, resample=Image.BICUBIC)
        
    raw_image = np.array(img)
    if len(raw_image.shape) == 2:
        raw_image = raw_image[:,:,None]
    else:
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    
    threshold_max = np.percentile(raw_image, 98)
    raw_image[raw_image > threshold_max] = threshold_max
    rescaled_image = cv2.convertScaleAbs(raw_image,
                                        alpha=(255.0 / raw_image.max()))

    # Perform histogram equalization.
    adjusted_image = cv2.equalizeHist(rescaled_image)
    img = Image.fromarray(adjusted_image.astype('uint8')).convert('RGB')
    return img


def clean_report(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    report = report.replace('..', '.').replace('. .', '.')
    return report


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
        else:
          token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    if len(bin_list) == 0 or len(bin_list) % 4 != 0:
        return None, False
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list, True


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip())) 
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


def construct_sample(image: Image, instruction: str, vqa=False):
    transformed_image = patch_resize_transform(image)
    if vqa:
        patch_image = torch.stack([transformed_image, transformed_image], dim=0)
        instruction_report = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True)
        vqa_instruction = 'what disease does this image have?'
        instruction_vqa = encode_text(' {}'.format(vqa_instruction.lower().strip()), append_bos=True, append_eos=True)
        instruction = collate_tokens([instruction_report, instruction_vqa], 0)
        patch_mask = torch.tensor([True, True])
    else:
        patch_image = transformed_image.unsqueeze(0)
        instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
        patch_mask = torch.tensor([True])
    
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def task_assgin(inst1, insts):
    best_sim = -1
    match_id = 0
    for i, inst2 in enumerate(insts):
        tokens_1 = gensim.utils.simple_preprocess(inst1)
        tokens_2 = gensim.utils.simple_preprocess(inst2)

        dictionary = corpora.Dictionary([tokens_1, tokens_2])

        corpus = [dictionary.doc2bow(tokens) for tokens in [tokens_1, tokens_2]]

        similarity = gensim.similarities.MatrixSimilarity(corpus)
        cosine_sim = similarity[corpus[0]][1]
        if cosine_sim > best_sim:
            best_sim = cosine_sim
            match_id = i

    if best_sim <= 0.05:
        return -1, match_id
    return best_sim, match_id


def resize_img(img, w, h, target=768):
    s = max(w, h)
    ratio = float(target) / s
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img = cv2.resize(img, (new_w, new_h))
    return img


def ask_answer(image, instruction):
    unkonwn = 'We apologize, but our model is unable to answer your question. Please feel free to try asking another question.'
    logger.info("Question: {}", instruction)
    
    cur_instruction = instruction.lower()
    detect_flag = False
    batch_report_vqa = False
    # image  = image_preprocess(image)
    w, h = image.size
    w_resize_ratio = task.cfg.patch_image_size / w
    h_resize_ratio = task.cfg.patch_image_size / h
    
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, cur_instruction, vqa=batch_report_vqa)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        
        if batch_report_vqa:
            tokens, bins, imgs = decode_fn(hypos[1][0]["tokens"], task.tgt_dict, task.bpe, generator)
            detected_disease = [dis.strip() for dis in tokens.split(',')]
            # if 'Support Devices' in detected_disease:
            #     detected_disease.remove('Support Devices')
            score = round(hypos[1][0]["score"].exp().item(), 6)
            logger.info("Detected disease: {}, {}", detected_disease, score)
            
        tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        score = round(hypos[0][0]["score"].exp().item(), 6)
        logger.info("Answer: {}, {}", score, tokens)
        logger.info("Coors: {}", bins)
        
    # tokens = tokens.replace('support devices', '')
    if len(tokens.split('.')) == 0 and not detect_flag:
        return unkonwn
    report = clean_report(tokens)
    report = report.split('.')
    report = [r.strip().capitalize() for r in report]
    report = ". ".join(report)
    report = report.replace('..', '.').replace('. .', '.')
    
    if detect_flag:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        for b, color in zip([bins], [(0, 229, 238)]):
            coord_list, bin_flag = bin2coord(b, w_resize_ratio, h_resize_ratio)
            if not bin_flag:
                return 'We are sorry, but we couldnt find any matches for this disease. Please try searching for another one.'
            cv2.rectangle(
                img,
                (int(coord_list[0]), int(coord_list[1])),
                (int(coord_list[2]), int(coord_list[3])),
                color,
                8
            )
            # cv2.putText(img, f'result', (int(coord_list[0]), int(coord_list[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0),
            #                                 thickness=2)
        out_name = './results/tmp/detect.jpg'
        img = resize_img(img, w, h)
        cv2.imwrite(out_name, img)
        return (out_name, report)
    return report