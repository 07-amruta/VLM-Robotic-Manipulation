import sys
import time

import cv2
from ultralytics import YOLOE

ROOT = '/workspaces/VLM-Robotic-Manipulation/'
image = cv2.imread(sys.argv[1])
model = YOLOE('yoloe-11s-seg.pt')
for name in sys.argv[2:]:
    model.set_classes([name], model.get_text_pe([name]))
    t = time.time()
    r = model.predict(image, conf=0.05, device='cuda:0', verbose=False)[0]
    n = 0 if r.boxes is None else len(r.boxes)
    found = [(r.boxes.xyxy[i].int().tolist(), round(float(r.boxes.conf[i]), 2)) for i in range(n)]
    print(name, 'n =', n, f'{time.time() - t:.2f}s', found)
    cv2.imwrite(ROOT + 'yoloe_' + name.replace(' ', '_') + '.png', r.plot())