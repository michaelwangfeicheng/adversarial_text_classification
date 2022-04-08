#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 13:46 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 13:46   wangfc      1.0         None
"""

from typing import Text, Optional
import json
import jsonschema


# from rasa.shared.exceptions import RasaException
class HsnlpException(Exception):
    """Base exception class for all errors raised by Rasa Open Source.

    These exceptions results from invalid use cases and will be reported
    to the users, but will be ignored in telemetry.
    """
    
    
class InvalidParameterException(HsnlpException, ValueError):
    """Raised when an invalid parameter is used."""


class YamlException(Exception):
    """Raised if there is an error reading yaml."""

    def __init__(self, filename: Optional[Text] = None) -> None:
        """Create exception.

        Args:
            filename: optional file the error occurred in"""
        self.filename = filename


class YamlSyntaxException(YamlException):
    """Raised when a YAML file can not be parsed properly due to a syntax error."""

    def __init__(
        self,
        filename: Optional[Text] = None,
        underlying_yaml_exception: Optional[Exception] = None,
    ) -> None:
        super(YamlSyntaxException, self).__init__(filename)

        self.underlying_yaml_exception = underlying_yaml_exception

    def __str__(self) -> Text:
        if self.filename:
            exception_text = f"Failed to read '{self.filename}'."
        else:
            exception_text = "Failed to read YAML."

        if self.underlying_yaml_exception:
            self.underlying_yaml_exception.warn = None
            self.underlying_yaml_exception.note = None
            exception_text += f" {self.underlying_yaml_exception}"

        if self.filename:
            exception_text = exception_text.replace(
                'in "<unicode string>"', f'in "{self.filename}"'
            )

        exception_text += (
            "\n\nYou can use https://yamlchecker.com/ to validate the "
            "YAML syntax of your file."
        )
        return exception_text



class MarkdownException(HsnlpException, ValueError):
    """Raised if there is an error reading Markdown."""




class FileNotFoundException(HsnlpException, FileNotFoundError):
    """Raised when a file, expected to exist, doesn't exist."""


class FileIOException(HsnlpException):
    """Raised if there is an error while doing file IO."""


class InvalidConfigException(ValueError, HsnlpException):
    """Raised if an invalid configuration is encountered."""


class UnsupportedFeatureException(HsnlpException):
    """Raised if a requested feature is not supported."""


class SchemaValidationError(HsnlpException, jsonschema.ValidationError):
    """Raised if schema validation via `jsonschema` failed."""


class InvalidEntityFormatException(HsnlpException, json.JSONDecodeError):
    """Raised if the format of an entity is invalid."""

    @classmethod
    def create_from(
        cls, other: json.JSONDecodeError, msg: Text
    ) -> "InvalidEntityFormatException":
        """Creates `InvalidEntityFormatException` from `JSONDecodeError`."""
        return cls(msg, other.doc, other.pos)


class ConnectionException(HsnlpException):
    """Raised when a connection to a 3rd party service fails.

    It's used by our broker and tracker store classes, when
    they can't connect to services like postgres, dynamoDB, mongo.
    """


class MissingArgumentError(ValueError):
    """Raised when not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message: Text) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> Text:
        return self.message


class UnsupportedLanguageError(Exception):
    """Raised when a component is created but the language is not supported.

    Attributes:
        component -- component name
        language -- language that component doesn't support
    """

    def __init__(self, component: Text, language: Text) -> None:
        self.component = component
        self.language = language

        super().__init__(component, language)

    def __str__(self) -> Text:
        return (
            f"component '{self.component}' does not support language '{self.language}'."
        )


class InvalidRuleException(Exception):
    """Exception that can be raised when rules are not valid."""

    def __init__(self, message: Text) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        return self.message



# from rasa.exceptions import ModelNotFound

class ModelNotFound(HsnlpException):
    """Raised when a model is not found in the path provided by the user."""


class NoEventsToMigrateError(HsnlpException):
    """Raised when no events to be migrated are found."""


class NoConversationsInTrackerStoreError(HsnlpException):
    """Raised when a tracker store does not contain any conversations."""


class NoEventsInTimeRangeError(HsnlpException):
    """Raised when a tracker store does not contain events within a given time range."""


class MissingDependencyException(HsnlpException):
    """Raised if a python package dependency is needed, but not installed."""


class PublishingError(HsnlpException):
    """Raised when publishing of an event fails.

    Attributes:
        timestamp -- Unix timestamp of the event during which publishing fails.
    """

    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp
        super(PublishingError, self).__init__()

    def __str__(self) -> Text:
        """Returns string representation of exception."""
        return str(self.timestamp)


# from rasa.nlu.registry import ComponentNotFoundException
class ComponentNotFoundException(ModuleNotFoundError, HsnlpException):
    """Raised if a module referenced by name can not be imported."""
    pass



# from rasa.nlu.model import  InvalidModelError
class InvalidModelError(HsnlpException):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message


class UnsupportedModelError(HsnlpException):
    """Raised when a model is too old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        self.message = message
        super(UnsupportedModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message
