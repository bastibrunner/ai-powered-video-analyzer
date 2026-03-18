from ultralytics import YOLO

    logging.info("Loading advanced YOLO model (YOLO11x)...")
    try:
        yolo_model = YOLO("yolo11x.pt")
        free_gpu()
    except Exception as e:
        logging.error("Error loading YOLO model: %s", str(e))
        return
