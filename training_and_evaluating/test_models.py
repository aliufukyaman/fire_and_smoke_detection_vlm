from typing import List, Tuple

import torch
from model_blip import ModelBlip
from model_clip import ModelClip
from PIL import Image
from utils import get_dataset_image_paths


def print_cuda_info() -> None:
    """
    Print CUDA availability and version information.
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")


def clear_cuda_cache() -> None:
    """
    Clear the CUDA cache to free up memory.
    """
    torch.cuda.empty_cache()


def test_models() -> None:
    """
    Test the BLIP and CLIP models on fire/smoke and non-fire/smoke images.
    """
    # Clear CUDA cache before starting
    clear_cuda_cache()

    # Get the dataset paths for both classes
    paths_firesmoke: List[str]
    paths_non_firesmoke: List[str]
    paths_firesmoke, paths_non_firesmoke = get_dataset_image_paths(
        "dataset/test/fire_or_smoke/", "dataset/test/non_fire_or_smoke"
    )

    # Define the models
    models: List[Tuple[str, object]] = [("blip", ModelBlip()), ("clip", ModelClip())]

    # Initialize counters for correct predictions
    correct_predictions_blip_firesmoke: int = 0
    correct_predictions_blip_non_firesmoke: int = 0
    correct_predictions_clip_firesmoke: int = 0
    correct_predictions_clip_non_firesmoke: int = 0

    counter: int = 0
    # Iterate over both image classes
    for image_class, paths in [
        ("firesmoke", paths_firesmoke),
        ("non_firesmoke", paths_non_firesmoke),
    ]:
        for image_path in paths:
            counter += 1
            print(f"{counter}: {image_path}")
            try:
                image: Image.Image = Image.open(image_path)

                # Set the correct label for the image class
                correct_label: bool = True if image_class == "firesmoke" else False

                for model_name, model in models:
                    result, caption = model.infer_image(image)
                    # Check if the prediction matches the expected label
                    if result == correct_label:
                        if model_name == "blip":  # BLIP model
                            if image_class == "firesmoke":
                                correct_predictions_blip_firesmoke += 1
                            else:
                                correct_predictions_blip_non_firesmoke += 1
                        elif model_name == "clip":  # CLIP model
                            if image_class == "firesmoke":
                                correct_predictions_clip_firesmoke += 1
                            else:
                                correct_predictions_clip_non_firesmoke += 1
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    # Calculate accuracy for each model and class
    accuracy_blip_firesmoke: float = (
        correct_predictions_blip_firesmoke / len(paths_firesmoke) * 100
    )
    accuracy_blip_non_firesmoke: float = (
        correct_predictions_blip_non_firesmoke / len(paths_non_firesmoke) * 100
    )
    accuracy_clip_firesmoke: float = (
        correct_predictions_clip_firesmoke / len(paths_firesmoke) * 100
    )
    accuracy_clip_non_firesmoke: float = (
        correct_predictions_clip_non_firesmoke / len(paths_non_firesmoke) * 100
    )

    # Print the results
    print(f"Accuracy for BLIP on firesmoke images: {accuracy_blip_firesmoke:.2f}%")
    print(
        f"Accuracy for BLIP on non_firesmoke images: {accuracy_blip_non_firesmoke:.2f}%"
    )
    print(f"Accuracy for CLIP on firesmoke images: {accuracy_clip_firesmoke:.2f}%")
    print(
        f"Accuracy for CLIP on non_firesmoke images: {accuracy_clip_non_firesmoke:.2f}%"
    )


if __name__ == "__main__":
    print_cuda_info()
    test_models()
