"""Custom Errors and Exceptions."""

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ('ArgumentError', 'ConfigError')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class ArgumentError(BaseException):
    """."""
    pass


class ConfigError(Exception):
    """Configuration Error."""

    def __init__(self, message=None, targets=None):
        """Dunder."""
        # Call the base class constructor with the parameters it needs
        if not message:
            message = 'Configuration Mismatch'
        super().__init__(message)

        # Now for your custom code...
        self.message = message
        self.params = targets

    def __repr__(self):
        """Representation Magic Method."""
        return self.message

# end of code

# end of file
