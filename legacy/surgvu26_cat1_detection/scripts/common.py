import numpy as np, cv2
NAMES=['grasping retractor','cadiere forceps','bipolar forceps','force bipolar','clip applier','stapler',
       'permanent cautery hook spatula','monopolar curved scissor','vessel sealer','tip up fenestrated grasper',
       'bipolar dissector','needle driver','prograsp forceps','suction irrigator']
GC_NAMES=[n.replace(' ','_') for n in NAMES]
UI_TOP=32          # rows 0..31 static UI (banner/letterbox)
UI_BOT=488         # rows 488..511 static UI (tool bar)
def mask_ui(img):
    """Black out the da Vinci UI bands. Hidden test set has these BLURRED; masking makes
    train/test consistent and prevents the model reading tool names from the UI."""
    img=img.copy(); img[:UI_TOP]=0; img[UI_BOT:]=0; return img
