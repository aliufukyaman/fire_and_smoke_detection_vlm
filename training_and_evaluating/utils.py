import os
import random
from typing import List, Tuple


def get_dataset_image_paths(
    folder_path_fire_smoke: str, folder_path_non_fire_smoke: str, count: int = None
) -> Tuple[List[str], List[str]]:
    """
    Get paths of images from the fire/smoke and non-fire/smoke folders.

    Args:
        folder_path_fire_smoke (str): Path to the fire/smoke images folder.
        folder_path_non_fire_smoke (str): Path to the non-fire/smoke images folder.
        count (int, optional): Maximum number of images to return from each folder. Defaults to None.

    Returns:
        Tuple[List[str], List[str]]: Lists of image paths for fire/smoke and non-fire/smoke.
    """
    image_extensions: Tuple[str, str, str] = (".jpg", ".jpeg", ".png")

    fire_smoke_images_path: List[str] = [
        os.path.join(folder_path_fire_smoke, file)
        for file in os.listdir(folder_path_fire_smoke)
        if file.lower().endswith(image_extensions)
    ]

    non_fire_smoke_images_path: List[str] = [
        os.path.join(folder_path_non_fire_smoke, file)
        for file in os.listdir(folder_path_non_fire_smoke)
        if file.lower().endswith(image_extensions)
    ]

    if count is not None:
        fire_smoke_images_path = fire_smoke_images_path[:count]
        non_fire_smoke_images_path = non_fire_smoke_images_path[:count]

    return fire_smoke_images_path, non_fire_smoke_images_path


class RandomImage:
    """
    Class to randomly select an image from the fire/smoke or non-fire/smoke dataset.
    """

    def __init__(self, folder_path_fire_smoke: str, folder_path_non_fire_smoke: str):
        """
        Initialize the RandomImage class.

        Args:
            folder_path_fire_smoke (str): Path to the fire/smoke images folder.
            folder_path_non_fire_smoke (str): Path to the non-fire/smoke images folder.
        """
        self.datasets: Tuple[List[str], List[str]] = get_dataset_image_paths(
            folder_path_fire_smoke, folder_path_non_fire_smoke
        )

    def get_random_image(self) -> Tuple[str, str]:
        """
        Get a random image from either the fire/smoke or non-fire/smoke dataset.

        Returns:
            Tuple[str, str]: Path to the image and its label ("firesmoke" or "nofiresmoke").
        """
        random_number: int = random.choice([0, 1])
        return (
            random.choice(self.datasets[random_number]),
            "firesmoke" if random_number == 0 else "nofiresmoke",
        )
