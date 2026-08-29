"""
ViT (Vision Transformer) model wrapper for CIFAR-10 using HuggingFace transformers.
Supports LoRA injection for parameter-efficient unlearning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, PeftModel


class ViTWrapper(nn.Module):
    """
    Wrapper for HuggingFace ViT model with CIFAR-10 adaptations.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 10,
        image_size: int = 224,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        
        if pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            config.image_size = image_size
            self.vit = ViTForImageClassification(config)
        
        # Store original classifier for reference
        self.original_classifier = self.vit.classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.vit(pixel_values=x)
        return outputs.logits
    
    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and [CLS] token hidden state."""
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        logits = outputs.logits
        # Last hidden state, [CLS] token (index 0)
        cls_hidden = outputs.hidden_states[-1][:, 0]
        return logits, cls_hidden
    
    def get_hidden_states(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get hidden states from a specific layer."""
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states[layer_idx]
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights from all layers."""
        outputs = self.vit(pixel_values=x, output_attentions=True)
        return outputs.attentions
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def attach_lora(
        self,
        rank: int = 16,
        alpha: float = 32,
        target_modules: List[str] = None,
        dropout: float = 0.05
    ):
        """Attach LoRA adapters to the ViT model."""
        if target_modules is None:
            target_modules = ["query", "key", "value", "dense"]
        
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="IMAGE_CLASSIFICATION"
        )
        
        self.vit = get_peft_model(self.vit, peft_config)
        return self.vit
    
    def merge_lora(self):
        """Merge LoRA weights into backbone."""
        if isinstance(self.vit, PeftModel):
            self.vit = self.vit.merge_and_unload()
    
    def print_trainable_params(self):
        """Print trainable parameter count."""
        if isinstance(self.vit, PeftModel):
            self.vit.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.vit.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.vit.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
        }
        if metadata:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> 'ViTWrapper':
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            model_name=checkpoint.get('model_name', 'google/vit-base-patch16-224'),
            num_classes=checkpoint.get('num_classes', 10),
            image_size=checkpoint.get('image_size', 224),
            pretrained=False
        )
        model.vit.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


def create_vit_model(
    model_name: str = "google/vit-base-patch16-224",
    num_classes: int = 10,
    image_size: int = 224,
    pretrained: bool = True,
    device: str = 'cuda'
) -> ViTWrapper:
    """Factory function to create ViT model."""
    model = ViTWrapper(model_name, num_classes, image_size, pretrained)
    return model.to(device)