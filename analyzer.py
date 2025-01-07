import io
import os
from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class ModelBlip:
    def __init__(self):
        """
        Initialize the BLIP model for image captioning and fire detection.
        """
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the BLIP processor and model
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        # Keywords to detect fire-related content in captions which works with highest accuracy
        self.keywords = [
            "fire",
            "smoke",
            "flame",
            "burning",
            "blaze",
            "explosion",
            "burn",
        ]

    def infer_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Infer the image and check if it contains fire-related content.

        Args:
            image (Image.Image): The image to analyze.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if fire is detected and the generated caption.
        """
        try:
            # Process the image and generate caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs, max_length=100, num_beams=1, repetition_penalty=2.0
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Check if any fire-related keyword is in the caption
            if any(keyword in caption.lower() for keyword in self.keywords):
                return True, caption.lower()
            else:
                return False, caption.lower()
        except Exception as e:
            # Handle any exceptions during inference
            raise RuntimeError(f"Error during image inference: {str(e)}")


class ModelBlipFineTuned:
    def __init__(self):
        """
        Initialize the fine-tuned BLIP model for fire and smoke detection.
        """
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "models/blip-fire-smoke-finetuned"
        self.is_model_exist = False

        # Check if the fine-tuned model exists
        if os.path.exists(model_path):
            self.is_model_exist = True

        # If model is not found, return empty
        if self.is_model_exist is False:
            return

        # Load the fine-tuned BLIP processor and model
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(
            self.device
        )

        # Predefined captions for fire/smoke and non-fire/smoke scenarios for finetuned model
        self.firesmoke_captions = [
            "a fire is burning with thick smoke.",
            "flames and smoke are visible in the image.",
            "there is a fire with heavy smoke.",
            "smoke is rising from a fire.",
        ]
        self.non_firesmoke_captions = [
            "no fire or smoke is present in the image.",
            "the scene is clear with no signs of fire or smoke.",
            "there is no fire or smoke in this image.",
            "everything looks normal with no fire or smoke.",
        ]

    def get_if_model_exist(self) -> bool:
        """
        Check if the fine-tuned model exists.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return self.is_model_exist

    def infer_image(self, image: Image.Image) -> Tuple[Optional[bool], Optional[str]]:
        """
        Infer the image using the fine-tuned model and check if it contains fire or smoke.

        Args:
            image (Image.Image): The image to analyze.

        Returns:
            Tuple[Optional[bool], Optional[str]]: A tuple containing a boolean indicating if fire/smoke is detected and the generated caption.
                                                  Returns (None, None) if the model does not exist or if the caption is not recognized.
        """
        try:
            if self.get_if_model_exist() is False:
                return None, "Finetuned model not found!"

            # Process the image and generate caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs, max_length=100, num_beams=1, repetition_penalty=2.0
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Check if the caption matches fire/smoke or non-fire/smoke scenarios
            if caption in self.firesmoke_captions:
                return True, caption.lower()
            elif caption in self.non_firesmoke_captions:
                return False, caption.lower()
            else:
                return None, None
        except Exception as e:
            # Handle any exceptions during inference
            raise RuntimeError(f"Error during image inference: {str(e)}")


class ImageAnalyzer:
    def __init__(self):
        """
        Initialize the image analyzer with both standard and fine-tuned models.
        """
        self.model_finetuned = ModelBlipFineTuned()
        self.model_standard = ModelBlip()

    def analyze(
        self, model_type: str, image: bytes
    ) -> Dict[str, Union[bool, Dict[str, Union[bool, str]], str]]:
        """
        Analyze the image using the selected model type.

        Args:
            model_type (str): The type of model to use ("finetuned" or "standard").
            image (bytes): The image data in bytes.

        Returns:
            Dict[str, Union[bool, Dict[str, Union[bool, str]], str]]: A dictionary containing the success status, analysis result, and error message if any.
        """
        try:
            # Open the image from bytes
            image = Image.open(io.BytesIO(image))

            # Analyze the image using the selected model
            if model_type == "finetuned":
                has_fire, caption = self.model_finetuned.infer_image(image)
            else:
                has_fire, caption = self.model_standard.infer_image(image)

            print(caption)

            # Return the analysis result to FastAPI to display on webpage
            return {
                "success": True,
                "analysis_result": {"has_fire": has_fire, "caption": caption},
            }

        except Exception as e:
            # Handle any exceptions during analysis
            return {"success": False, "error": str(e)}
