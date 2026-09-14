class MissingToolCallError(Exception):
    """Raised when a required tool call is missing in the response."""

    pass


class MalformedArgumentsError(Exception):
    """Raised when the arguments in the tool call are malformed."""

    pass
