# Ultralytics YOLO 🚀, GPL-3.0 license

# import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops, LOGGER
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2
# from paddleocr import PaddleOCR
import os
import re
from ultralytics.yolo.configs import get_config
import yaml
from datetime import datetime, timezone
from io import BytesIO
import PIL.Image as Image
# from LPRNet.predict import OcrPredictor
import math
from collections import Counter


# ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=0, show_log=False)

def distance_between_points(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.sqrt(dx ** 2 + dy ** 2)

class TrackBlob:
    def __init__(self, bbox, txt, full_img, crop_img):
        self.bbox = [bbox]
        self.centerPosition = [( (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 )]
        self.dblCurrentDiagonalSize = [math.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2)]
        self.machine_number = [txt]
        self.full_img = [full_img]
        self.crop_img = [crop_img]
        self.frame_num = 0

    def update(self, bbox, txt, full_img, crop_img):
        self.bbox.append(bbox)
        self.centerPosition.append(( (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 ))
        self.dblCurrentDiagonalSize.append(math.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2))
        self.machine_number.append(txt)
        self.full_img.append(full_img)
        self.crop_img.append(crop_img)
        self.frame_num = 0

ocr_filter = re.compile("^(영)?(((강원|경기|경남|경북|광주|대구|대전|부산|서울|인천|전남|전북|제주|충남|충북)\d{2})|(\d{3})|(\d{2}))(가|나|다|라|마|거|너|더|러|머|버|서|어|저|고|노|도|로|모|보|소|오|조|구|누|두|루|무|부|수|우|주|기|니|디|리|미|비|시|이|지|바|사|아|자|카|파|타|차|배|영|하|허|호|히|육|공|해|국|합)\d{4}")
class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch, queue):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # ocr_result_manage
        self.send_ocr_result(queue)

        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            imc = im0.copy()
            self.ocr_image(imc, xyxy, conf)
            # with self.dt[3]:
            #     self.ocr_image(imc, xyxy, conf)
            # LOGGER.info(f"ocr:{self.dt[3].dt * 1E3:.1f}ms")
            
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string
    
    def send_ocr_result(self, queue):
        if len(self.existed_blob) == 0:
            return

        entry_direction = os.getenv("entry_direction", default="left")
        del_index = []
        for idx, blob in enumerate(self.existed_blob):
            most_common_string=None
            blob.frame_num = blob.frame_num + 1
            if blob.frame_num > 15:
                print("FRAME::15")
                for mn in blob.machine_number:
                    if ocr_filter.match(mn) is not None:
                        most_common_string=mn
                        break
                if most_common_string is None:
                    counter = Counter(blob.machine_number)
                    most_common_string, _ = counter.most_common(1)[0]
                print("most_common_string", most_common_string)

                targetIndex = blob.machine_number.index(most_common_string)
                now = datetime.now(timezone.utc).timestamp()

                direction = ""
                if len(blob.centerPosition) > 0:
                    direction = "left" if blob.centerPosition[0][0] > blob.centerPosition[0][1] else "right"
                else:
                    direction = "left"
                
                datekey = "entry_date" if entry_direction == direction else "exit_date"

                img_io_crop = BytesIO()
                img_io_full = BytesIO()
                Image.fromarray(blob.crop_img[targetIndex]).save(img_io_crop, format="PNG")
                Image.fromarray(blob.full_img[targetIndex]).save(img_io_full, format="PNG")
                img_bytes_crop = img_io_crop.getvalue()
                img_bytes_full = img_io_full.getvalue()
                
                result_fetch = {"machine_number": most_common_string.replace("영",""),
                                "full_image": img_bytes_full,
                                "cropped_image": img_bytes_crop,
                                datekey: now
                                }
                del_index.append(idx)
                print("SEND::result->queue")
                # queue.put_nowait(result_fetch)
            else:
                continue
        
        for x in del_index:
            del(self.existed_blob[x])

    def ocr_image(self, img, cordinates, conf):
        det_score = conf.item()
        # print("det_score", det_score)
        if det_score < 0.6:
            return

        x1, y1, x2, y2 = int(cordinates[0]), int(cordinates[1]), int(cordinates[2]), int(cordinates[3])
        img_crop = img[y1:y2, x1:x2]
        img_full_RGB = img[:, :, ::-1]
        img_crop_RGB = img_crop[:, :, ::-1]

        result = self.ocr.predict(img_crop)
        # print("ocr::result", result)
        # print("ocr::text", result[0])
        if len(self.existed_blob) == 0:
            blob = TrackBlob([x1, y1, x2, y2], result[0], img_full_RGB, img_crop_RGB)
            self.existed_blob.append(blob)
        else:
            flag=False
            for blob in self.existed_blob:
                currentCenterPosition = ((x1 + x2) / 2, (y1 + y2) / 2)
                existedCenterPosition = blob.centerPosition[-1]
                distance = distance_between_points(currentCenterPosition, existedCenterPosition)

                if distance < blob.dblCurrentDiagonalSize[-1]:
                    # print("UPDATE::BLOB")
                    blob.update([x1, y1, x2, y2], result[0], img_full_RGB, img_crop_RGB)
                    flag=True
                    break
            if not flag:
                new_blob = TrackBlob([x1, y1, x2, y2], result[0], img_full_RGB, img_crop_RGB)
                self.existed_blob.append(new_blob)
        
        # ---------------------------paddleocr----------------------------------
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # for x in result[0]:
        #     # print(x[1][0])
        #     # print(x[1][1])
        #     if self.ocr_result is None and 4 <= len(str(x[1][0])) <= 8:
        #         self.ocr_result = {"score": int(x[1][1]), "text": x[1][0], "frame_num": 0, "frame_img": img }
        #         continue

        #     if 4 <= len(str(x[1][0])) <= 8 and self.ocr_result["score"] < int(x[1][1]):
        #         self.ocr_result = {"score": int(x[1][1]), "text": x[1][0], "frame_num": 0, "frame_img": img }

# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def set_config(cfg):
#     cfg.model = './runs/detect/model_trained/weights/best.pt'
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size

#     # rtsp = os.getenv("rtsp")
#     rtsp = './ultralytics/yolo/v8/detect/stream_piece_0.mp4'

#     if rtsp is None:
#         raise Exception("'rtsp' and 'rtsp2' are required")

#     # cfg.source = os.getenv("rtsp")
#     cfg.source = rtsp
#     cfg.device = 'cuda:0'
#     cfg.show = True
#     cfg.save = False
#     # print("asfasf", cfg)
#     overrides = {}
#     result = get_config(cfg, overrides)
#     print("result", result)
#     return result

# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(mp_queue):
    with open(DEFAULT_CONFIG,encoding="utf8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg["model"] = './runs/detect/model_trained_epoch_150/weights/best.pt'
    cfg["imgsz"] = check_imgsz(cfg["imgsz"], min_dim=2)  # check image size
 
    # rtsp = os.getenv("rtsp")
    rtsp = 'test_images/stream_piece_0.mp4'
    # rtsp = 'rtsp://wg2b659c:adt@6400@10.172.177.3:554/h264'
    # rtsp = "rtsp://admin:adt@2102@223.48.2.77:554/cam/realmonitor?channel=1&subtype=1"
    # rtsp = "./test12.png"
    # rtsp = "entry_1728273193_full.png"
 
    if rtsp is None:
        raise Exception("'rtsp' and 'rtsp2' are required")
 
    # cfg["source = os.getenv("rtsp")
    cfg["source"] = rtsp
    cfg["device"] = 'cuda:0'
    cfg["show"] = True
    cfg["save"] = False

    predictor = DetectionPredictor(cfg)
    predictor(queue=mp_queue)
