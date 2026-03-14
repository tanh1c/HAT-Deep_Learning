"""
Model Registry - Central place to register and manage all models.

This module makes it easy to add new models for different datasets.
Each model handler should implement the BaseModelHandler interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image


class PredictionResult:
    """Container for prediction results from a model."""

    def __init__(
        self,
        label: str,
        confidence: float,
        all_labels: List[str],
        all_confidences: List[float],
        explanation_image: Optional[np.ndarray] = None,
    ):
        self.label = label
        self.confidence = confidence
        self.all_labels = all_labels
        self.all_confidences = all_confidences
        self.explanation_image = explanation_image  # Grad-CAM or attention map


class CalibrationResult:
    """Container for model calibration analysis results."""

    def __init__(
        self,
        ece: float,
        bin_accuracies: List[float],
        bin_confidences: List[float],
        bin_counts: List[int],
        reliability_diagram: Optional[Any] = None,
        source: Optional[str] = None,
    ):
        self.ece = ece
        self.bin_accuracies = bin_accuracies
        self.bin_confidences = bin_confidences
        self.bin_counts = bin_counts
        self.reliability_diagram = reliability_diagram
        self.source = source


class BaseModelHandler(ABC):
    """
    Abstract base class for model handlers.
    
    To add a new model, create a subclass and implement all abstract methods.
    Then register it in the MODEL_REGISTRY dictionary below.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset name this model was trained on."""
        pass

    @abstractmethod
    def get_data_type(self) -> str:
        """Return data type: 'image', 'text', or 'multimodal'."""
        pass

    @abstractmethod
    def get_class_labels(self) -> List[str]:
        """Return list of class labels."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Return dict of model info for display (architecture, params, etc.)."""
        pass

    @abstractmethod
    def predict(self, input_data) -> PredictionResult:
        """
        Run prediction on input data.
        
        For image models: input_data is a PIL Image or numpy array
        For text models: input_data is a string
        For multimodal: input_data is a tuple (image, text)
        
        Returns: PredictionResult
        """
        pass

    @abstractmethod
    def get_example_inputs(self) -> List[Any]:
        """Return list of example inputs for the demo."""
        pass

    def get_calibration_data(
        self, max_samples: Optional[int] = None
    ) -> Optional[CalibrationResult]:
        """
        Optionally return calibration analysis result.
        Override this in subclass if you want calibration display.
        """
        return None


# Global model registry - add new models here
MODEL_REGISTRY: Dict[str, BaseModelHandler] = {}


def register_model(key: str, handler: BaseModelHandler):
    """Register a model handler in the global registry."""
    MODEL_REGISTRY[key] = handler


def get_model_handler(key: str) -> Optional[BaseModelHandler]:
    """Get a model handler by key."""
    return MODEL_REGISTRY.get(key)


def get_all_model_keys() -> List[str]:
    """Get all registered model keys."""
    return list(MODEL_REGISTRY.keys())


def get_models_by_type(data_type: str) -> Dict[str, BaseModelHandler]:
    """Get all models of a specific data type."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.get_data_type() == data_type}
