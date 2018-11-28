"""Custom Errors and Exceptions."""


class ConfigError(Exception):
    """Configuration Error."""

    def __init__(self, message, errors):
        """Initialization Magic Method."""
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.message = message
        self.errors = errors

    def __repr__(self):
        """Representation Magic Method."""
        return self.message

# end of file
