import os
import random
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor

# This file is used for finetuning the pre-trained Blip model for fire and smoke
# detection with costom dataset and captions.

# Define the main data directory
data_dir: str = "dataset"

# Define subdirectories for train, validation, and test datasets
train_dir: str = os.path.join(data_dir, "train")
val_dir: str = os.path.join(data_dir, "val")
test_dir: str = os.path.join(data_dir, "test")


def generate_caption(label: str) -> str:
    """
    Generate a random caption based on the label for finetuning purpose.

    Args:
        label (str): The label of the image, either "fire_smoke" or "non_fire_smoke".

    Returns:
        str: A randomly selected caption based on the label.
    """
    if label == "fire_smoke":
        captions: List[str] = [
            "A fire is burning with thick smoke.",
            "Flames and smoke are visible in the image.",
            "There is a fire with heavy smoke.",
            "Smoke is rising from a fire.",
        ]
    else:
        captions: List[str] = [
            "No fire or smoke is present in the image.",
            "The scene is clear with no signs of fire or smoke.",
            "There is no fire or smoke in this image.",
            "Everything looks normal with no fire or smoke.",
        ]
    return random.choice(captions)


class FireSmokeDataset(Dataset):
    """
    Custom dataset class for fire and smoke images.
    """

    def __init__(self, data_dir: str, processor: BlipProcessor):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the images.
            processor (BlipProcessor): BLIP processor for image and text processing.
        """
        self.data_dir: str = data_dir
        self.processor: BlipProcessor = processor
        self.image_paths: List[str] = []
        self.captions: List[str] = []

        # Load images and captions from the directory
        for label in ["fire_smoke", "non_fire_smoke"]:
            label_dir: str = os.path.join(data_dir, label)
            for image_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, image_name))
                self.captions.append(generate_caption(label))

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its corresponding caption by index.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed image and caption tensors.
        """
        image_path: str = self.image_paths[idx]
        caption: str = self.captions[idx]
        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            return inputs.pixel_values.squeeze(0), inputs.input_ids.squeeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.tensor([]), torch.tensor([])


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad images and captions to the same size.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of image and caption tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Stacked and padded image and caption tensors.
    """
    pixel_values: List[torch.Tensor] = [item[0] for item in batch]
    input_ids: List[torch.Tensor] = [item[1] for item in batch]

    # Stack images to the same size
    pixel_values: torch.Tensor = torch.stack(pixel_values)

    # Pad captions to the same size
    input_ids: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )

    return pixel_values, input_ids


# Load BLIP model and processor
model_name: str = "Salesforce/blip-image-captioning-base"
processor: BlipProcessor = BlipProcessor.from_pretrained(model_name)
model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(
    model_name
)

# Move model to GPU if available
device: str = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create datasets and DataLoaders
train_dataset: FireSmokeDataset = FireSmokeDataset(train_dir, processor)
val_dataset: FireSmokeDataset = FireSmokeDataset(val_dir, processor)
test_dataset: FireSmokeDataset = FireSmokeDataset(test_dir, processor)

train_loader: DataLoader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
)
val_loader: DataLoader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)
test_loader: DataLoader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)

# Define optimizer and loss function
optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()


def train(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: torch.optim.AdamW,
    criterion: torch.nn.CrossEntropyLoss,
    num_epochs: int = 5,
) -> None:
    """
    Train the model with custom dataset.

    Args:
        model (BlipForConditionalGeneration): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.AdamW): Optimizer for training.
        criterion (torch.nn.CrossEntropyLoss): Loss function.
        num_epochs (int): Number of epochs to train.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss: float = 0.0
        for pixel_values, input_ids in dataloader:
            pixel_values, input_ids = pixel_values.to(device), input_ids.to(device)

            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values, input_ids=input_ids, labels=input_ids
            )
            loss: torch.Tensor = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}"
        )


def validate(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
) -> None:
    """
    Validate the model.

    Args:
        model (BlipForConditionalGeneration): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.CrossEntropyLoss): Loss function.
    """
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for pixel_values, input_ids in dataloader:
            pixel_values, input_ids = pixel_values.to(device), input_ids.to(device)
            outputs = model(
                pixel_values=pixel_values, input_ids=input_ids, labels=input_ids
            )
            val_loss += outputs.loss.item()

    print(f"Validation Loss: {val_loss/len(dataloader):.4f}")


# Fine-tuning process
num_epochs: int = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, train_loader, optimizer, criterion)
    validate(model, val_loader, criterion)

# Save the fine-tuned model and processor
model.save_pretrained("blip-fire-smoke-finetuned")
processor.save_pretrained("blip-fire-smoke-finetuned")
