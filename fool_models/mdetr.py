import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from skimage.measure import find_contours

from matplotlib.patches import Polygon

torch.set_grad_enabled(False)

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
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, scores, boxes, labels, masks=None):
    def apply_mask(image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    plt.figure(figsize=(16, 10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
        masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

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

    plt.imshow(np_image)
    plt.axis('off')
    plt.show()


def plot_inference_qa(model, im, caption, plot=False):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

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

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)
            labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]
            plot_results(im, probas[keep], bboxes_scaled, labels)

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
    print(f"Predicted answer: {answers}")
    return answers

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_qa = torch.hub.load('ashkamath/mdetr:main', 'mdetr_clevr', pretrained=True, return_postprocessor=False)
model_qa = model_qa.cuda()
model_qa.eval()

pathlib.PosixPath = temp

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZXXghtIqfQZlIGuZiIwOB2ihlytB0ytEZ5p2u1p6LBbK0cDhKydplOUXmOWQukgIE1Zo&usqp=CAU"
im3 = Image.open(requests.get(url, stream=True).raw)

plot_inference_qa(model_qa, im3,
                  "There is a small sphere in front of the purple shiny sphere, in front of the large shiny object; What is its color?")
