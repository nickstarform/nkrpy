"""Custom Errors and Exceptions."""


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

# end of file
