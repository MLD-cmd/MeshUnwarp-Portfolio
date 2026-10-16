MIN_DIM = 288    # minimum working dimension; small inputs get upscaled
MAX_DIM = 2400   # maximum working edge; huge phone photos get downscaled

DOCTR_INPUT_SIZE    = 288 # GeoTr expects the image resized to 288x288 before predicting the warp grid.
GRID_FLOW_DENOM     = 286.8 # Used to normalize GeoTr’s predicted grid values into the coordinate range PyTorch expects, roughly -1 to 1.
GRID_RANGE_SHRINK   = 0.99 # Slightly shrinks the predicted grid range so the warp does not sample too aggressively outside the image.
ILLTR_BATCH         = 64 #llTr processes the image in 128x128 patches. This controls how many patches are processed at once.


WARP_STRENGTH       = 0.35 # Controls how much of GeoTr’s warp is applied in the center of the document.
EDGE_MARGIN         = 0.22    # fade zone: outermost 22% of each side
EDGE_WARP_STRENGTH  = 0.08   # warp strength at the very border
CHAOS_THRESHOLD     = 0.0050 # If the predicted warp field is too chaotic/noisy, the program reduces the warp strength.

VISION_MIN_CONF          = 0.3 # Minimum confidence for keeping OCR text boxes from Apple Vision.
VISION_TARGET_SHORT_DIM  = 1400 # Before OCR, the image is resized so the shorter side targets 1400 px. This gives OCR enough resolution to read text
VISION_MAX_SHORT_DIM     = 2400 # Upper limit for OCR scaling. Prevents OCR images from becoming too large and slow.
VISION_FAST_ACCEPT_CHARS = 140 # If Apple Vision already extracts at least 140 non-space characters from the raw image, the program accepts that OCR result quickly.


EASY_TARGET_SHORT_DIM    = 1600 # EasyOCR gets a slightly larger target size than Vision.

# These constants control image resizing, dewarp strength, illumination patch batching, and OCR input quality so the pipeline stays accurate without becoming too slow or unstable.
