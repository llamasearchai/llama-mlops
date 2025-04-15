# File: llama_mlops/tests/test_feature_store.py
"""
Unit tests for the FeatureStoreClient.
"""

import unittest
from unittest.mock import patch

import numpy as np

from llama_mlops.feature_store import FeatureStoreClient


class TestFeatureStoreClient(unittest.TestCase):
    """
    Tests for the FeatureStoreClient class.
    """

    @patch("llama_mlops.feature_store.get_env_variable")
    def setUp(self, mock_get_env_variable):
        """
        Set up test fixtures.
        """
        mock_get_env_variable.return_value = "test-api-key"
        self.client = FeatureStoreClient()

    def test_initialization(self):
        """
        Test initialization of the client.
        """
        self.assertEqual(self.client.api_key, "test-api-key")
        self.assertEqual(self.client.store, {})

    def test_put_and_get_feature(self):
        """
        Test putting and getting a feature.
        """
        feature_id = "test_feature"
        feature_data = {
            "name": "Test Feature",
            "values": {"entity1": 1.0, "entity2": 2.0},
        }

        # Put feature
        self.client.put_feature(feature_id, feature_data)

        # Get feature
        retrieved_feature = self.client.get_feature(feature_id)

        self.assertEqual(retrieved_feature, feature_data)

    def test_get_feature_not_found(self):
        """
        Test getting a non-existent feature.
        """
        with self.assertRaises(KeyError):
            self.client.get_feature("non_existent_feature")

    def test_get_feature_vector(self):
        """
        Test getting a feature vector.
        """
        # Add multiple features
        self.client.put_feature(
            "feature1", {"name": "Feature 1", "values": {"entity1": 1.0}}
        )
        self.client.put_feature(
            "feature2", {"name": "Feature 2", "values": {"entity1": 2.0}}
        )

        # Get feature vector
        feature_vector = self.client.get_feature_vector(["feature1", "feature2"])

        self.assertEqual(len(feature_vector), 2)
        self.assertEqual(feature_vector["feature1"]["name"], "Feature 1")
        self.assertEqual(feature_vector["feature2"]["name"], "Feature 2")

    def test_get_training_data(self):
        """
        Test getting training data.
        """
        # Add features with values
        self.client.put_feature(
            "feature1",
            {
                "name": "Feature 1",
                "values": {"entity1": 1.0, "entity2": 2.0, "entity3": 3.0},
            },
        )
        self.client.put_feature(
            "feature2",
            {
                "name": "Feature 2",
                "values": {"entity1": 4.0, "entity2": 5.0, "entity3": 6.0},
            },
        )

        # Get training data with all entities
        df = self.client.get_training_data(["feature1", "feature2"])

        self.assertEqual(len(df), 3)  # 3 entities
        self.assertEqual(len(df.columns), 2)  # 2 features

        # Get training data with specific entities
        df_subset = self.client.get_training_data(
            feature_ids=["feature1", "feature2"], entity_ids=["entity1", "entity3"]
        )

        self.assertEqual(len(df_subset), 2)  # 2 entities
        self.assertEqual(list(df_subset.index), ["entity1", "entity3"])

        # Verify values
        self.assertEqual(df_subset.loc["entity1", "feature1"], 1.0)
        self.assertEqual(df_subset.loc["entity3", "feature2"], 6.0)


# File: llama_mlops/tests/test_model_registry.py
"""
Unit tests for the ModelRegistry.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from llama_mlops.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """
    Tests for the ModelRegistry class.
    """

    @patch("llama_mlops.model_registry.get_env_variable")
    def setUp(self, mock_get_env_variable):
        """
        Set up test fixtures.
        """
        mock_get_env_variable.return_value = "test-api-key"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        self.registry = ModelRegistry(storage_path=self.storage_path)

    def tearDown(self):
        """
        Tear down test fixtures.
        """
        self.temp_dir.cleanup()

    def test_initialization(self):
        """
        Test initialization of the registry.
        """
        self.assertEqual(self.registry.api_key, "test-api-key")
        self.assertEqual(self.registry.storage_path, self.storage_path)
        self.assertEqual(self.registry.registry, {})
        self.assertTrue(os.path.isdir(self.storage_path))

    def test_register_and_get_model(self):
        """
        Test registering and getting a model.
        """
        # Create a test model file
        model_path = os.path.join(self.storage_path, "test_model.mlx")
        with open(model_path, "w") as f:
            f.write("Test model content")

        # Register model
        metadata = {"name": "Test Model", "version": "1.0.0"}
        model_id = self.registry.register_model(
            model_path=model_path, metadata=metadata, model_id="test-model-id"
        )

        # Get model
        model_entry = self.registry.get_model(model_id)

        self.assertEqual(model_id, "test-model-id")
        self.assertEqual(model_entry["model_path"], model_path)
        self.assertEqual(model_entry["metadata"], metadata)
        self.assertEqual(model_entry["status"], "registered")

    def test_update_model_status(self):
        """
        Test updating model status.
        """
        # Register model
        model_path = os.path.join(self.storage_path, "test_model.mlx")
        with open(model_path, "w") as f:
            f.write("Test model content")

        model_id = self.registry.register_model(
            model_path=model_path,
            metadata={"name": "Test Model"},
            model_id="test-model-id",
        )

        # Update status
        status_metadata = {"reason": "Testing", "user": "test_user"}
        self.registry.update_model_status(
            model_id=model_id, status="deployed", status_metadata=status_metadata
        )

        # Get updated model
        model_entry = self.registry.get_model(model_id)

        self.assertEqual(model_entry["status"], "deployed")
        self.assertIn("status_history", model_entry)
        self.assertEqual(len(model_entry["status_history"]), 1)
        self.assertEqual(model_entry["status_history"][0]["status"], "deployed")
        self.assertEqual(model_entry["status_history"][0]["metadata"], status_metadata)

    def test_list_models(self):
        """
        Test listing models.
        """
        # Register multiple models
        model_path = os.path.join(self.storage_path, "test_model.mlx")
        with open(model_path, "w") as f:
            f.write("Test model content")

        self.registry.register_model(
            model_path=model_path,
            metadata={"name": "Model 1", "type": "classifier"},
            model_id="model-1",
        )

        self.registry.register_model(
            model_path=model_path,
            metadata={"name": "Model 2", "type": "regressor"},
            model_id="model-2",
        )

        # List all models
        models = self.registry.list_models()
        self.assertEqual(len(models), 2)

        # List with filter
        classifiers = self.registry.list_models(filter_by={"type": "classifier"})
        self.assertEqual(len(classifiers), 1)
        self.assertEqual(classifiers[0]["model_id"], "model-1")

    def test_delete_model(self):
        """
        Test deleting a model.
        """
        # Register model
        model_path = os.path.join(self.storage_path, "test_model.mlx")
        with open(model_path, "w") as f:
            f.write("Test model content")

        model_id = self.registry.register_model(
            model_path=model_path,
            metadata={"name": "Test Model"},
            model_id="test-model-id",
        )

        # Delete model
        self.registry.delete_model(model_id)

        # Check that model is deleted
        with self.assertRaises(KeyError):
            self.registry.get_model(model_id)


# File: llama_mlops/tests/test_mlx_optimization.py
"""
Unit tests for the MLXOptimizer.
"""

import unittest
from unittest.mock import patch

from llama_mlops.constants import (
    OPTIMIZATION_LEVEL_AGGRESSIVE,
    OPTIMIZATION_LEVEL_BASIC,
    OPTIMIZATION_LEVEL_NONE,
)
from llama_mlops.mlx_optimization import MLXOptimizer


class TestMLXOptimizer(unittest.TestCase):
    """
    Tests for the MLXOptimizer class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.optimizer = MLXOptimizer()

    @patch("llama_mlops.mlx_optimization.HAS_MLX", False)
    def test_initialization_without_mlx(self):
        """
        Test initialization without MLX available.
        """
        optimizer = MLXOptimizer(optimization_level=OPTIMIZATION_LEVEL_BASIC)
        self.assertEqual(optimizer.optimization_level, OPTIMIZATION_LEVEL_BASIC)

    def test_optimize_none_level(self):
        """
        Test optimization with none level.
        """
        self.optimizer.optimization_level = OPTIMIZATION_LEVEL_NONE
        model = "test_model"

        result = self.optimizer.optimize(model)

        # With none level, model should be returned unchanged
        self.assertEqual(result, model)

    @patch("llama_mlops.mlx_optimization.MLXOptimizer.compile")
    def test_optimize_basic_level(self, mock_compile):
        """
        Test optimization with basic level.
        """
        mock_compile.return_value = "compiled_model"

        self.optimizer.optimization_level = OPTIMIZATION_LEVEL_BASIC
        model = "test_model"

        result = self.optimizer.optimize(model)

        # With basic level, model should be compiled
        mock_compile.assert_called_once_with(model, optimize_for_inference=True)
        self.assertEqual(result, "compiled_model")

    @patch("llama_mlops.mlx_optimization.MLXOptimizer.compile")
    @patch("llama_mlops.mlx_optimization.MLXOptimizer.quantize")
    def test_optimize_aggressive_level(self, mock_quantize, mock_compile):
        """
        Test optimization with aggressive level.
        """
        mock_quantize.return_value = "quantized_model"
        mock_compile.return_value = "compiled_quantized_model"

        self.optimizer.optimization_level = OPTIMIZATION_LEVEL_AGGRESSIVE
        model = "test_model"

        result = self.optimizer.optimize(model)

        # With aggressive level, model should be quantized and compiled
        mock_quantize.assert_called_once_with(model, bits=8, skip_modules=None)
        mock_compile.assert_called_once_with(
            "quantized_model", optimize_for_inference=True
        )
        self.assertEqual(result, "compiled_quantized_model")

    @patch("llama_mlops.mlx_optimization.HAS_MLX", False)
    def test_benchmark_simulation(self):
        """
        Test benchmarking simulation.
        """
        import numpy as np

        model = "test_model"
        input_data = {"input": np.random.rand(1, 3, 224, 224).astype(np.float32)}

        results = self.optimizer.benchmark(model, input_data)

        self.assertIn("avg_latency_ms", results)
        self.assertIn("min_latency_ms", results)
        self.assertIn("max_latency_ms", results)
        self.assertIn("p90_latency_ms", results)


# File: llama_mlops/tests/test_tee_service.py
"""
Unit tests for the TEEService.
"""

import unittest
from unittest.mock import patch

from llama_mlops.tee_service import TEEService


class TestTEEService(unittest.TestCase):
    """
    Tests for the TEEService class.
    """

    @patch("llama_mlops.tee_service.get_env_variable")
    def setUp(self, mock_get_env_variable):
        """
        Set up test fixtures.
        """
        mock_get_env_variable.return_value = "test-api-key"
        self.tee_service = TEEService()

    def test_initialization(self):
        """
        Test initialization of the service.
        """
        self.assertEqual(self.tee_service.api_key, "test-api-key")

    def test_encrypt_decrypt_data_bytes(self):
        """
        Test encrypting and decrypting data as bytes.
        """
        test_data = b"Test data for encryption"

        encrypted = self.tee_service.encrypt_data(test_data)
        decrypted = self.tee_service.decrypt_data(encrypted)

        self.assertEqual(decrypted, test_data)

    def test_encrypt_decrypt_data_dict(self):
        """
        Test encrypting and decrypting data as dictionary.
        """
        test_data = {"key": "value", "nested": {"list": [1, 2, 3]}}

        encrypted = self.tee_service.encrypt_data(test_data)
        decrypted = self.tee_service.decrypt_data(encrypted)

        # Decrypted data will be bytes of the JSON string
        decrypted_dict = json.loads(decrypted.decode("utf-8"))
        self.assertEqual(decrypted_dict, test_data)

    def test_encrypt_decrypt_data_numpy(self):
        """
        Test encrypting and decrypting data as NumPy array.
        """
        test_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        encrypted = self.tee_service.encrypt_data(test_data)
        decrypted = self.tee_service.decrypt_data(encrypted)

        # Convert back to NumPy array for comparison
        decrypted_array = np.frombuffer(decrypted, dtype=np.float32)
        np.testing.assert_array_equal(decrypted_array, test_data)

    def test_secure_train(self):
        """
        Test secure training.
        """
        model_data = b"Test model data"
        training_data = b"Test training data"
        config_data = json.dumps({"epochs": 10, "batch_size": 32}).encode("utf-8")

        encrypted_model = self.tee_service.encrypt_data(model_data)
        encrypted_data = self.tee_service.encrypt_data(training_data)
        encrypted_config = self.tee_service.encrypt_data(config_data)

        encrypted_trained_model = self.tee_service.secure_train(
            encrypted_model=encrypted_model,
            encrypted_data=encrypted_data,
            encrypted_config=encrypted_config,
        )

        # Decrypt and verify
        trained_model = self.tee_service.decrypt_data(encrypted_trained_model)

        # In our simulation, the trained model is the original model with "_trained" appended
        self.assertEqual(trained_model, model_data + b"_trained")

    def test_secure_inference(self):
        """
        Test secure inference.
        """
        model_data = b"Test model data"
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32).tobytes()

        encrypted_model = self.tee_service.encrypt_data(model_data)
        encrypted_input = self.tee_service.encrypt_data(input_data)

        encrypted_output = self.tee_service.secure_inference(
            encrypted_model=encrypted_model, encrypted_input=encrypted_input
        )

        # Decrypt output
        output = self.tee_service.decrypt_data(encrypted_output)

        # In our simulation, the output is a random array of 10 floats
        output_array = np.frombuffer(output, dtype=np.float32)
        self.assertEqual(len(output_array), 10)

    def test_get_verify_attestation(self):
        """
        Test getting and verifying attestation.
        """
        attestation = self.tee_service.get_attestation()

        # Verify valid attestation
        is_valid = self.tee_service.verify_attestation(attestation)
        self.assertTrue(is_valid)

        # Verify invalid attestation
        invalid_attestation = attestation + b"invalid"
        is_valid = self.tee_service.verify_attestation(invalid_attestation)
        self.assertFalse(is_valid)


# Updated in commit 6 - 2025-04-04 17:44:56

# Updated in commit 14 - 2025-04-04 17:44:57

# Updated in commit 22 - 2025-04-04 17:44:57

# Updated in commit 30 - 2025-04-04 17:44:59

# Updated in commit 6 - 2025-04-05 14:44:07

# Updated in commit 14 - 2025-04-05 14:44:07

# Updated in commit 22 - 2025-04-05 14:44:07

# Updated in commit 30 - 2025-04-05 14:44:07

# Updated in commit 6 - 2025-04-05 15:30:18

# Updated in commit 14 - 2025-04-05 15:30:18

# Updated in commit 22 - 2025-04-05 15:30:19

# Updated in commit 30 - 2025-04-05 15:30:19

# Updated in commit 6 - 2025-04-05 16:10:23

# Updated in commit 14 - 2025-04-05 16:10:23

# Updated in commit 22 - 2025-04-05 16:10:24

# Updated in commit 30 - 2025-04-05 16:10:24

# Updated in commit 6 - 2025-04-05 17:17:03

# Updated in commit 14 - 2025-04-05 17:17:03

# Updated in commit 22 - 2025-04-05 17:17:03

# Updated in commit 30 - 2025-04-05 17:17:03

# Updated in commit 6 - 2025-04-05 17:48:37

# Updated in commit 14 - 2025-04-05 17:48:37

# Updated in commit 22 - 2025-04-05 17:48:37

# Updated in commit 30 - 2025-04-05 17:48:37

# Updated in commit 6 - 2025-04-05 18:38:19

# Updated in commit 14 - 2025-04-05 18:38:19

# Updated in commit 22 - 2025-04-05 18:38:19

# Updated in commit 30 - 2025-04-05 18:38:19

# Updated in commit 6 - 2025-04-05 18:50:03

# Updated in commit 14 - 2025-04-05 18:50:03

# Updated in commit 22 - 2025-04-05 18:50:03

# Updated in commit 30 - 2025-04-05 18:50:03

# Updated in commit 6 - 2025-04-05 19:12:19

# Updated in commit 14 - 2025-04-05 19:12:19

# Updated in commit 22 - 2025-04-05 19:12:20

# Updated in commit 30 - 2025-04-05 19:12:20
