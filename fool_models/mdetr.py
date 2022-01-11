import json
import os
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.patches import Polygon
from skimage.measure import find_contours

from neural_render.blender_render_utils.constants import find_platform_slash, find_platform

PLATFORM_SLASH = find_platform_slash()
UP_TO_HERE_ = PLATFORM_SLASH.join(os.path.abspath(__file__).split(PLATFORM_SLASH)[:-2]).replace(PLATFORM_SLASH, '/')


def invert_dict(d):
    return {value: key for (key, value) in d.items()}


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

with open(f'{UP_TO_HERE_}/fool_models/resources/vocab.json', 'r') as fin:
    data = json.loads(fin.read())

question_token_to_idx = data['question_token_to_idx']
program_token_to_idx = data['program_token_to_idx']
answer_token_to_idx = data['answer_token_to_idx']

idx_to_question_token = invert_dict(question_token_to_idx)
idx_to_program_token = invert_dict(program_token_to_idx)
idx_to_answer_token = invert_dict(answer_token_to_idx)

data['question_idx_to_token'] = idx_to_question_token
data['program_idx_to_token'] = idx_to_program_token
data['answer_idx_to_token'] = idx_to_answer_token

ALL_ATTRIBUTES = [
    "small",
    "large",
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "purple",
    "cyan",
    "yellow",
    "cube",
    "sphere",
    "cylinder",
    "rubber",
    "metal",
]


class MDetrWrapper:
    def __init__(self):
        self.model = None
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                       [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        self.transform = T.Compose([
            T.Resize(700),
            # T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def plot_results(self, pil_img, scores, boxes, labels, masks=None):
        def apply_mask(image, mask, color, alpha=0.5):
            """Apply the given mask to the image.
            """
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          image[:, :, c])
            return image

        plt.figure(figsize=(16, 16))
        np_image = np.array(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        if masks is None:
            masks = [None for _ in range(len(scores))]
        assert len(scores) == len(boxes) == len(labels) == len(masks)
        for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{l}: {s:0.2f}'
            if 'red' in text:
                xmin -= 60
            ax.text(xmin, ymin, text, fontsize=30, bbox=dict(facecolor='white', alpha=1))

            if mask is None:
                continue
            np_image = apply_mask(np_image, mask, c)

            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=c)
                ax.add_patch(p)

        np_image = np_image.transpose(2, 0, 1)[None]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        np_image = (np_image * std) + mean
        np_image = np_image[0].transpose(1, 2, 0)[None]
        plt.axis('off')
        plt.imshow(np_image[0])
        plt.show()


    def load_mdetr(self):
        if find_platform() == 'WIN':
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        model_qa = torch.hub.load('ashkamath/mdetr:main', 'mdetr_clevr', pretrained=True, return_postprocessor=False)
        model_qa = model_qa.cuda()
        model_qa.eval()
        if find_platform() == 'WIN':
            pathlib.PosixPath = temp
        self.model = model_qa
        return self.model

    def inference(self, im, caption, plot=False):
        if self.model is None:
            self.load_mdetr()
        img = self.transform(im).unsqueeze(0).cuda()

        # propagate through the model
        memory_cache = self.model(img, [caption], encode_and_save=True)
        outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        if plot:
            # keep only predictions with 0.7+ confidence
            probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
            keep = (probas > 0.7).cpu()

            positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
            predicted_spans = defaultdict(str)
            for tok in positive_tokens:
                item, pos = tok
                if pos < 255:
                    span = memory_cache["tokenized"].token_to_chars(0, pos)
                    predicted_spans[item] += " " + caption[span.start:span.end]

            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], (224, 224))
            labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]
            self.plot_results(im.permute(1, 2, 0).cpu(), probas[keep].detach().cpu(), bboxes_scaled.detach(), labels)

        answer_types = outputs["pred_answer_type"].argmax(-1)
        answer_types = [x.item() for x in answer_types]
        answers = None
        for i, ans_type in enumerate(answer_types):
            if ans_type == 0:
                answers = "yes" if outputs["pred_answer_binary"][i].sigmoid() > 0.5 else "no"
            elif ans_type == 1:
                answers = ALL_ATTRIBUTES[outputs["pred_answer_attr"][i].argmax(-1).item()]
            elif ans_type == 2:
                answers = str(outputs["pred_answer_reg"][i].argmax(-1).item())
            else:
                assert False, "must be one of the answer types"
        return answers

    def eval(self):
        return

    def to(self, where):
        return

    def __call__(self, image, question, plot=False):
        ImageBatchSize = image.size(0)
        QuestionBatchSize = question.size(0)
        assert ImageBatchSize == QuestionBatchSize
        # Images will be propagated as they are #
        # Questions will be first converted from indexes to tokens for compatibility #
        answers = []
        for index in range(ImageBatchSize):
            image_ = image[index, :]
            question_ = question[index, :].cpu().numpy()
            restored_question = ' '.join([idx_to_question_token[f] for f in question_ if f not in [1, 2, 0]])
            answer = self.inference(im=image_, caption=restored_question, plot=plot)
            restored_answer = answer_token_to_idx[answer]
            answers.append(restored_answer)

        torch_answers = torch.LongTensor(answers)
        return torch_answers, None, None
