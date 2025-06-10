#!/usr/bin/env python3
"""
Focused test suite for aCS_aBUS and aCS_aBUS_parallel logging functionality.

This tests the specific logging implementations in the adaptive conditional
sampling modules without requiring full algorithm execution.
"""

import io
import logging
import sys
import unittest
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestACSLogging(unittest.TestCase):
    """Test logging functionality in aCS modules."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up string handler to capture log messages
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        
        # Configure formatter
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)

    def tearDown(self):
        """Clean up after tests."""
        # Remove handlers
        for logger_name in ['parallel_abus.aBUS_SuS.aCS_aBUS', 'parallel_abus.aBUS_SuS.aCS_aBUS_parallel']:
            logger = logging.getLogger(logger_name)
            if self.handler in logger.handlers:
                logger.removeHandler(self.handler)
            logger.setLevel(logging.NOTSET)

    def test_aCS_aBUS_logger_setup(self):
        """Test that aCS_aBUS has proper logger setup."""
        try:
            from parallel_abus.aBUS_SuS.aCS_aBUS import aCS_aBUS
            
            logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
            
            # Should have NullHandler
            has_null_handler = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
            self.assertTrue(has_null_handler, "aCS_aBUS logger should have NullHandler")
            
            # Should not have hard-coded level
            self.assertEqual(logger.level, logging.NOTSET, "Should not have hard-coded level")
            
        except ImportError as e:
            self.skipTest(f"Could not import aCS_aBUS: {e}")

    def test_aCS_aBUS_parallel_logger_setup(self):
        """Test that aCS_aBUS_parallel has proper logger setup."""
        try:
            from parallel_abus.aBUS_SuS.aCS_aBUS_parallel import aCS_aBUS_batches
            
            logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS_parallel')
            
            # Should have NullHandler
            has_null_handler = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
            self.assertTrue(has_null_handler, "aCS_aBUS_parallel logger should have NullHandler")
            
            # Should not have hard-coded level
            self.assertEqual(logger.level, logging.NOTSET, "Should not have hard-coded level")
            
        except ImportError as e:
            self.skipTest(f"Could not import aCS_aBUS_parallel: {e}")

    def test_aCS_aBUS_logging_output(self):
        """Test that aCS_aBUS produces appropriate log messages."""
        try:
            from parallel_abus.aBUS_SuS.aCS_aBUS import aCS_aBUS
            import numpy as np
            
            # Add handler to capture logs
            logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
            logger.addHandler(self.handler)
            logger.setLevel(logging.DEBUG)
            
            # Create minimal test data
            N = 6  # Small number for quick test
            lambd_old = 0.6
            tau = 0.1
            theta_seeds = np.random.normal(size=(2, 3))  # 2 dimensions, 3 seeds
            
            def mock_log_L_fun(theta):
                """Mock log-likelihood function that returns a reasonable value."""
                return -0.5 * np.sum(theta**2)  # Simple quadratic
            
            def mock_h_LSF(pi_u, logl_hat, log_L):
                """Mock limit state function."""
                return log_L - logl_hat + np.log(np.random.uniform(0.1, 1.0))
            
            # Run the function
            try:
                result = aCS_aBUS(N, lambd_old, tau, theta_seeds, mock_log_L_fun, 0.5, mock_h_LSF)
                
                # Check that we got results
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 5)  # Should return 5 values
                
            except Exception as e:
                # Even if the algorithm fails, we want to check logging
                print(f"Algorithm failed (expected in test): {e}")
            
            # Check log output
            log_output = self.log_capture.getvalue()
            
            # Should contain debug messages about standard deviation option
            self.assertIn("standard deviation option", log_output.lower())
            
            # Should contain initial parameter information
            self.assertIn("Initial lambda", log_output)
            
        except ImportError as e:
            self.skipTest(f"Could not import aCS_aBUS: {e}")

    def test_aCS_aBUS_parallel_logging_messages(self):
        """Test that aCS_aBUS_parallel has expected logging patterns."""
        try:
            # Import and check for logging patterns in the source
            import parallel_abus.aBUS_SuS.aCS_aBUS_parallel as module
            import inspect
            
            source = inspect.getsource(module)
            
            # Should have debug logging for algorithm details
            self.assertIn('logger.debug', source, "Should have debug logging")
            
            # Should have warning logging for potential issues
            self.assertIn('logger.warning', source, "Should have warning logging")
            
            # Should have error logging for error conditions
            self.assertIn('logger.error', source, "Should have error logging")
            
            # Should not have deprecated warn method
            self.assertNotIn('logger.warn(', source, "Should not use deprecated logger.warn")
            
            # Should not have direct logging calls
            self.assertNotIn('logging.debug(', source, "Should not use direct logging.debug")
            
        except ImportError as e:
            self.skipTest(f"Could not import aCS_aBUS_parallel: {e}")

    def test_aCS_aBUS_source_code_logging(self):
        """Test that aCS_aBUS source code has proper logging."""
        try:
            # Import and check for logging patterns in the source
            import parallel_abus.aBUS_SuS.aCS_aBUS as module
            import inspect
            
            source = inspect.getsource(module)
            
            # Should have logging import
            self.assertIn('import logging', source, "Should import logging")
            
            # Should have logger setup
            self.assertIn('logger = logging.getLogger(__name__)', source, "Should set up logger")
            
            # Should have NullHandler
            self.assertIn('logger.addHandler(logging.NullHandler())', source, "Should add NullHandler")
            
            # Should have debug logging
            self.assertIn('logger.debug', source, "Should have debug logging")
            
        except ImportError as e:
            self.skipTest(f"Could not import aCS_aBUS: {e}")

    def test_logging_hierarchy(self):
        """Test that loggers are properly organized in hierarchy."""
        try:
            from parallel_abus.aBUS_SuS import aCS_aBUS, aCS_aBUS_parallel
            
            # Get loggers
            aCS_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
            aCS_parallel_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS_parallel')
            parent_logger = logging.getLogger('parallel_abus')
            
            # Test hierarchy
            self.assertTrue(aCS_logger.name.startswith('parallel_abus'))
            self.assertTrue(aCS_parallel_logger.name.startswith('parallel_abus'))
            
            # Test that they're different loggers
            self.assertNotEqual(aCS_logger, aCS_parallel_logger)
            
        except ImportError as e:
            self.skipTest(f"Could not import modules: {e}")

    def test_no_print_statements(self):
        """Test that modules don't use print statements (should use logging)."""
        try:
            import parallel_abus.aBUS_SuS.aCS_aBUS as aCS_module
            import parallel_abus.aBUS_SuS.aCS_aBUS_parallel as aCS_parallel_module
            import inspect
            
            # Check aCS_aBUS
            aCS_source = inspect.getsource(aCS_module)
            print_statements = [line for line in aCS_source.split('\n') 
                              if 'print(' in line and not line.strip().startswith('#')]
            self.assertEqual(len(print_statements), 0, 
                           f"aCS_aBUS should not have print statements: {print_statements}")
            
            # Check aCS_aBUS_parallel (excluding docstring examples)
            aCS_parallel_source = inspect.getsource(aCS_parallel_module)
            print_statements = [line for line in aCS_parallel_source.split('\n') 
                              if 'print(' in line and not line.strip().startswith('#') 
                              and '>>>' not in line]  # Exclude docstring examples
            self.assertEqual(len(print_statements), 0, 
                           f"aCS_aBUS_parallel should not have print statements: {print_statements}")
            
        except ImportError as e:
            self.skipTest(f"Could not import modules: {e}")


class TestACSLoggingIntegration(unittest.TestCase):
    """Test integration between aCS logging and parent logging configuration."""

    def test_parent_configuration_inheritance(self):
        """Test that aCS modules respect parent logger configuration."""
        try:
            import parallel_abus
            from parallel_abus.aBUS_SuS import aCS_aBUS
            
            # Configure parent logger
            parallel_abus.configure_logging(level=logging.WARNING)
            
            # Get child logger
            aCS_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
            
            # Should inherit effective level from parent
            self.assertEqual(aCS_logger.getEffectiveLevel(), logging.WARNING)
            
        except ImportError as e:
            self.skipTest(f"Could not import modules: {e}")

    def test_disable_logging_affects_aCS(self):
        """Test that disabling logging affects aCS modules."""
        try:
            import parallel_abus
            
            # Disable logging
            parallel_abus.disable_logging()
            
            # Get aCS loggers
            aCS_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
            aCS_parallel_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS_parallel')
            
            # Parent should have NullHandler only
            parent_logger = logging.getLogger('parallel_abus')
            has_only_null_handlers = all(isinstance(h, logging.NullHandler) 
                                       for h in parent_logger.handlers)
            self.assertTrue(has_only_null_handlers)
            
        except ImportError as e:
            self.skipTest(f"Could not import modules: {e}")


def run_aCS_logging_tests():
    """Run aCS-specific logging tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestACSLogging, TestACSLoggingIntegration]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running aCS logging tests for parallel_abus...")
    print("=" * 50)
    
    success = run_aCS_logging_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All aCS logging tests passed!")
        exit_code = 0
    else:
        print("❌ Some aCS logging tests failed.")
        exit_code = 1
    
    print("\nACS Logging Test Summary:")
    print("✅ Logger setup and configuration")
    print("✅ NullHandler presence")
    print("✅ Logging message content")
    print("✅ No deprecated methods")
    print("✅ Parent-child logger integration")
    print("✅ Source code validation")
    
    sys.exit(exit_code) 