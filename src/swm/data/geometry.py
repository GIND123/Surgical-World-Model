"""Image geometry shared by every stage: the 1280x720 SurgVU frame, the 640x512 cache and the model input.

The cache keeps the legacy detector geometry (pillarbox removed, console UI bands kept so the robot-state
readout can be computed); the world model only ever sees the endoscope field, rows FIELD_ROWS of the cache.
"""

SRC_W, SRC_H = 1280, 720
SRC_CROP = (192, 1088, 4, 716)          # x0, x1, y0, y1 in the source frame -> 896 x 712 (even, so
                                        # 4:2:0 chroma crops align; legacy detector used 193-1087 x 4-717)
CACHE_W, CACHE_H = 640, 512
CACHE_FPS = 10.0
FIELD_ROWS = (38, 480)                  # endoscope field inside the cache (442 x 640): source rows 56-672;
                                        # the banner ends at 55 and a UI shadow band covers 673-681
UI_TOP_ROWS = (0, 32)
UI_BOTTOM_ROWS = (488, 512)
MODEL_H, MODEL_W = 224, 320             # FIELD resized (x0.507 vertical, x0.5 horizontal)
