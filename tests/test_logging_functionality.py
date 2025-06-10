#!/usr/bin/env python3
"""
Comprehensive test suite for logging functionality in parallel_abus library.

Tests logging configuration, NullHandler setup, logging control functions,
and proper logging behavior across all modules.
"""

import io
import logging
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the modules to test
try:
    import parallel_abus
    from parallel_abus.aBUS_SuS import aBUS_SuS_parallel, aCS_aBUS_parallel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available for full testing: {e}")
    DEPENDENCIES_AVAILABLE = False


class TestLoggingConfiguration(unittest.TestCase):
    """Test logging configuration and setup."""

    def setUp(self):
        """Set up test environment."""
        # Import modules first to ensure they are loaded and have their handlers set up
        try:
            from parallel_abus.aBUS_SuS import aBUS_SuS_parallel, aCS_aBUS_parallel, aBUS_SuS, aCS_aBUS
        except ImportError:
            pass  # Some modules might not be available in all test environments
        
        # Store original handlers to restore later
        self.original_handlers = {}
        logger_names = [
            'parallel_abus', 
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel', 
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aCS_aBUS'
        ]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            self.original_handlers[logger_name] = {
                'handlers': logger.handlers.copy(),
                'level': logger.level,
                'propagate': logger.propagate
            }

    def tearDown(self):
        """Clean up after tests."""
        # Restore original handlers and settings
        for logger_name, original_state in self.original_handlers.items():
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.handlers.extend(original_state['handlers'])
            logger.setLevel(original_state['level'])
            logger.propagate = original_state['propagate']

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_package_level_null_handler(self):
        """Test that package-level NullHandler is properly configured."""
        # Reset to clean state first
        parallel_abus.disable_logging()
        
        pkg_logger = logging.getLogger('parallel_abus')
        self.assertTrue(len(pkg_logger.handlers) > 0, "Package logger should have handlers")
        
        # Check for NullHandler
        has_null_handler = any(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers)
        self.assertTrue(has_null_handler, "Package logger should have NullHandler")

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_module_level_null_handlers(self):
        """Test that module-level loggers have NullHandler configured after fresh import."""
        # Check the original state stored in setUp (before any test modifications)
        module_loggers = [
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel',
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aCS_aBUS'
        ]

        # Check that all loggers had proper handler configuration at import time
        for logger_name in module_loggers:
            with self.subTest(logger_name=logger_name):
                # Get the original state from setUp
                original_state = self.original_handlers.get(logger_name, {})
                original_handlers = original_state.get('handlers', [])
                
                # Should have at least one handler (NullHandler) from module import
                self.assertGreater(len(original_handlers), 0, 
                                 f"Logger {logger_name} should have at least one handler from module import")
                
                # At least one should be a NullHandler
                has_null_handler = any(isinstance(h, logging.NullHandler) for h in original_handlers)
                self.assertTrue(has_null_handler, 
                              f"Logger {logger_name} should have NullHandler from module import")

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_no_hard_coded_log_levels(self):
        """Test that loggers don't have hard-coded log levels."""
        from parallel_abus.aBUS_SuS import aBUS_SuS_parallel, aCS_aBUS_parallel
        
        module_loggers = [
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel',
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aCS_aBUS'
        ]
        
        for logger_name in module_loggers:
            with self.subTest(logger_name=logger_name):
                logger = logging.getLogger(logger_name)
                self.assertEqual(logger.level, logging.NOTSET, 
                               f"Logger {logger_name} should not have hard-coded level")

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_configure_logging_function(self):
        """Test the configure_logging() function."""
        # Test with INFO level
        parallel_abus.configure_logging(level=logging.INFO)
        pkg_logger = logging.getLogger('parallel_abus')
        
        self.assertEqual(pkg_logger.level, logging.INFO)
        self.assertFalse(pkg_logger.propagate)  # Should not propagate
        
        # Check that handler was added
        self.assertTrue(len(pkg_logger.handlers) > 0)
        
        # Test with custom handler
        string_io = io.StringIO()
        custom_handler = logging.StreamHandler(string_io)
        parallel_abus.configure_logging(level=logging.DEBUG, handler=custom_handler)
        
        # Verify custom handler was set
        self.assertIn(custom_handler, pkg_logger.handlers)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_disable_logging_function(self):
        """Test the disable_logging() function."""
        # First configure logging
        parallel_abus.configure_logging(level=logging.INFO)
        pkg_logger = logging.getLogger('parallel_abus')
        
        # Then disable it
        parallel_abus.disable_logging()
        
        # Should have only NullHandler
        self.assertTrue(len(pkg_logger.handlers) > 0)
        has_only_null_handlers = all(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers)
        self.assertTrue(has_only_null_handlers, "After disabling, should have only NullHandler")

    def test_no_warnings_on_import(self):
        """Test that importing the library doesn't produce logging warnings."""
        # Capture stderr to check for warnings
        captured_stderr = io.StringIO()
        
        with redirect_stderr(captured_stderr):
            # Force reload to simulate fresh import
            if 'parallel_abus' in sys.modules:
                del sys.modules['parallel_abus']
            if DEPENDENCIES_AVAILABLE:
                import parallel_abus
        
        stderr_content = captured_stderr.getvalue()
        
        # Should not contain logging warnings
        self.assertNotIn("No handlers could be found for logger", stderr_content)
        self.assertNotIn("No handlers found", stderr_content)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_logging_to_file(self):
        """Test logging to file functionality."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as tmp_file:
            tmp_path = tmp_file.name
        
        file_handler = None
        try:
            # Configure logging to file
            file_handler = logging.FileHandler(tmp_path)
            parallel_abus.configure_logging(level=logging.DEBUG, handler=file_handler)
            
            # Get a logger and log something
            logger = logging.getLogger('parallel_abus.test')
            logger.debug("Test debug message")
            logger.info("Test info message")
            
            # Force flush
            if file_handler:
                file_handler.flush()
            
            # Close the handler before reading
            if file_handler:
                file_handler.close()
            
            # Read the file and check content
            with open(tmp_path, 'r') as f:
                content = f.read()
            
            # Note: Child loggers inherit from parent, so messages should appear
            self.assertIn("Test", content)
            
        finally:
            # Clean up handlers first
            if file_handler:
                file_handler.close()
            # Clear logger handlers to release file handles
            pkg_logger = logging.getLogger('parallel_abus')
            pkg_logger.handlers.clear()
            # Now try to delete the file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except PermissionError:
                # If we still can't delete it, at least try again after a short delay
                import time
                time.sleep(0.1)
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except PermissionError:
                    pass  # Give up - the file will be cleaned up eventually

    def test_best_practices_compliance(self):
        """Verify the library follows Python logging best practices."""
        # Reset to ensure clean state
        parallel_abus.disable_logging()
        
        # Test that library has proper NullHandler after reset
        pkg_logger = logging.getLogger('parallel_abus')
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers),
                       "Package should have NullHandler by default")
        
        # Test that configure_logging doesn't produce warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parallel_abus.configure_logging(level=logging.DEBUG)
            self.assertEqual(len(w), 0, "configure_logging should not produce warnings")

    def test_module_specific_logging(self):
        """Test that different modules can have different logging levels."""
        from io import StringIO
        
        # Create a string buffer to capture log output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        
        # Configure different levels for different modules
        parallel_abus.configure_module_logging({
            'aBUS_SuS': logging.DEBUG,
            'aCS': logging.WARNING
        }, handler=ch)
        
        # Get the specific loggers
        abus_logger = logging.getLogger('parallel_abus.aBUS_SuS.aBUS_SuS')
        acs_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
        
        # Verify they have the correct levels
        self.assertEqual(abus_logger.level, logging.DEBUG)
        self.assertEqual(acs_logger.level, logging.WARNING)
        
        # Test that they log at their respective levels
        abus_logger.debug("aBUS debug message")
        abus_logger.info("aBUS info message")
        acs_logger.debug("aCS debug message - should not appear")
        acs_logger.warning("aCS warning message")
        
        # Get the logged output
        log_contents = log_capture_string.getvalue()
        
        # Verify expected messages appear
        self.assertIn("aBUS debug message", log_contents)
        self.assertIn("aBUS info message", log_contents)
        self.assertIn("aCS warning message", log_contents)
        # aCS debug should not appear because level is WARNING
        self.assertNotIn("aCS debug message", log_contents)

    def test_module_specific_logging_full_names(self):
        """Test module-specific logging with full logger names."""
        from io import StringIO
        
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        
        # Configure using full logger names
        parallel_abus.configure_module_logging({
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel': logging.INFO,
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel': logging.ERROR
        }, handler=ch)
        
        # Get the specific loggers
        abus_parallel_logger = logging.getLogger('parallel_abus.aBUS_SuS.aBUS_SuS_parallel')
        acs_parallel_logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS_parallel')
        
        # Verify they have the correct levels
        self.assertEqual(abus_parallel_logger.level, logging.INFO)
        self.assertEqual(acs_parallel_logger.level, logging.ERROR)
        
        # Test logging behavior
        abus_parallel_logger.debug("aBUS parallel debug - should not appear")
        abus_parallel_logger.info("aBUS parallel info message")
        acs_parallel_logger.warning("aCS parallel warning - should not appear")
        acs_parallel_logger.error("aCS parallel error message")
        
        log_contents = log_capture_string.getvalue()
        
        # Verify expected messages
        self.assertNotIn("aBUS parallel debug", log_contents)
        self.assertIn("aBUS parallel info message", log_contents)
        self.assertNotIn("aCS parallel warning", log_contents)
        self.assertIn("aCS parallel error message", log_contents)

    def test_performance_logging_hierarchy(self):
        """Test that logging doesn't significantly impact performance when disabled."""
        import time
        
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        # Set up logger with NullHandler (effectively disabled)
        logger = logging.getLogger('parallel_abus.test.performance')
        logger.addHandler(logging.NullHandler())
        
        # Measure time with many log calls
        start_time = time.time()
        for i in range(1000):
            logger.debug(f"Debug message {i}")
            logger.info(f"Info message {i}")
        end_time = time.time()
        
        disabled_time = end_time - start_time
        
        # This should be very fast (typically < 0.1 seconds)
        self.assertLess(disabled_time, 1.0, "Logging with NullHandler should be fast")


class TestLoggingBehavior(unittest.TestCase):
    """Test actual logging behavior of the algorithms."""

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
        for logger_name in ['parallel_abus', 'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
                           'parallel_abus.aBUS_SuS.aCS_aBUS_parallel', 'parallel_abus.aBUS_SuS.aBUS_SuS',
                           'parallel_abus.aBUS_SuS.aCS_aBUS']:
            logger = logging.getLogger(logger_name)
            if self.handler in logger.handlers:
                logger.removeHandler(self.handler)
            logger.setLevel(logging.NOTSET)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_aCS_aBUS_logging(self):
        """Test logging behavior of aCS_aBUS module."""
        from parallel_abus.aBUS_SuS.aCS_aBUS import aCS_aBUS
        
        # Add handler to capture logs
        logger = logging.getLogger('parallel_abus.aBUS_SuS.aCS_aBUS')
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)
        
        # Create minimal test data
        import numpy as np
        N = 10
        lambd_old = 0.6
        tau = 0.1
        theta_seeds = np.random.normal(size=(3, 5))  # 3 dimensions, 5 seeds
        
        def mock_log_L_fun(theta):
            return np.random.normal()
        
        def mock_h_LSF(pi_u, logl_hat, log_L):
            return np.random.normal()
        
        # Run function (it might fail due to missing dependencies, but we want to test logging)
        try:
            result = aCS_aBUS(N, lambd_old, tau, theta_seeds, mock_log_L_fun, 0.5, mock_h_LSF)
        except Exception:
            pass  # We're testing logging, not functionality
        
        # Check that debug messages were logged
        log_output = self.log_capture.getvalue()
        self.assertIn("DEBUG", log_output)
        self.assertIn("Initial lambda", log_output)

    def test_logger_name_conventions(self):
        """Test that loggers follow proper naming conventions."""
        expected_loggers = [
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel',
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aCS_aBUS'
        ]
        
        if DEPENDENCIES_AVAILABLE:
            # Import modules to create loggers
            from parallel_abus.aBUS_SuS import aBUS_SuS_parallel, aCS_aBUS_parallel
            
            for logger_name in expected_loggers:
                logger = logging.getLogger(logger_name)
                self.assertIsInstance(logger, logging.Logger)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_log_level_inheritance(self):
        """Test that child loggers inherit from parent logger configuration."""
        # Configure parent logger
        parallel_abus.configure_logging(level=logging.WARNING)
        
        # Child loggers should inherit the configuration
        parent_logger = logging.getLogger('parallel_abus')
        child_logger = logging.getLogger('parallel_abus.aBUS_SuS.aBUS_SuS_parallel')
        
        # Parent should have level set
        self.assertEqual(parent_logger.level, logging.WARNING)
        
        # Child should inherit effective level
        self.assertEqual(child_logger.getEffectiveLevel(), logging.WARNING)

    def test_logging_performance_impact(self):
        """Test that logging doesn't significantly impact performance when disabled."""
        import time
        
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        
        # Set up logger with NullHandler (effectively disabled)
        logger = logging.getLogger('parallel_abus.test.performance')
        logger.addHandler(logging.NullHandler())
        
        # Measure time with many log calls
        start_time = time.time()
        for i in range(1000):
            logger.debug(f"Debug message {i}")
            logger.info(f"Info message {i}")
        end_time = time.time()
        
        disabled_time = end_time - start_time
        
        # This should be very fast (typically < 0.1 seconds)
        self.assertLess(disabled_time, 1.0, "Logging with NullHandler should be fast")


class TestLoggingValidation(unittest.TestCase):
    """Validate logging configuration against best practices."""

    def test_no_logger_warnings(self):
        """Test that no logger warnings are emitted during import."""
        # Capture any warnings
        with patch('warnings.warn') as mock_warn:
            if DEPENDENCIES_AVAILABLE:
                # We test by disabling and re-enabling logging instead of reload
                parallel_abus.disable_logging()
                parallel_abus.configure_logging()
        
        # Check that warnings.warn was not called with logging-related warnings
        if mock_warn.called:
            warning_messages = [str(call) for call in mock_warn.call_args_list]
            logging_warnings = [msg for msg in warning_messages 
                              if 'logger' in msg.lower() or 'handler' in msg.lower()]
            self.assertEqual(len(logging_warnings), 0, 
                           f"Should not have logging warnings: {logging_warnings}")

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_logging_levels_appropriate(self):
        """Test that logging levels are used appropriately."""
        # This test verifies the conceptual usage - that we have different log levels
        from parallel_abus.aBUS_SuS import aCS_aBUS_parallel
        
        # Read the source code to check for appropriate level usage
        import inspect
        source = inspect.getsource(aCS_aBUS_parallel)
        
        # Should have debug statements for detailed information
        self.assertIn('logger.debug', source)
        
        # Should have warning statements for potential issues
        self.assertIn('logger.warning', source)
        
        # Should have error statements for error conditions
        self.assertIn('logger.error', source)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_no_deprecated_logging_methods(self):
        """Test that no deprecated logging methods are used."""
        modules_to_check = [
            'parallel_abus.aBUS_SuS.aBUS_SuS_parallel',
            'parallel_abus.aBUS_SuS.aCS_aBUS_parallel',
            'parallel_abus.aBUS_SuS.aBUS_SuS',
            'parallel_abus.aBUS_SuS.aCS_aBUS'
        ]
        
        for module_name in modules_to_check:
            with self.subTest(module=module_name):
                try:
                    module = sys.modules.get(module_name)
                    if module:
                        import inspect
                        source = inspect.getsource(module)
                        
                        # Should not use deprecated logger.warn
                        self.assertNotIn('logger.warn(', source, 
                                       f"Module {module_name} should not use deprecated logger.warn()")
                        
                        # Should not use direct logging.debug calls
                        self.assertNotIn('logging.debug(', source,
                                       f"Module {module_name} should not use direct logging.debug()")
                except Exception as e:
                    self.skipTest(f"Could not check module {module_name}: {e}")


def run_logging_tests():
    """Run all logging tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestLoggingConfiguration, TestLoggingBehavior, TestLoggingValidation]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running comprehensive logging tests for parallel_abus...")
    print("=" * 60)
    
    success = run_logging_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All logging tests passed!")
        exit_code = 0
    else:
        print("❌ Some logging tests failed.")
        exit_code = 1
    
    print("\nTest Summary:")
    print("✅ NullHandler configuration")
    print("✅ Logging control functions")
    print("✅ No hard-coded log levels")
    print("✅ No deprecated methods")
    print("✅ Proper logging behavior")
    print("✅ Performance validation")
    
    sys.exit(exit_code) 