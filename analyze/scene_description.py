from transformers import BlipProcessor, BlipForConditionalGeneration

    logging.info("Loading BLIP-2 captioning model (base variant)...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(device)
        free_gpu()
    except Exception as e:
        logging.error("Error loading BLIP model: %s", str(e))
        return
