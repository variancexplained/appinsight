#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : AppInsight                                                                          #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.3                                                                              #
# Filename   : /appinsight/utils/repo.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/appinsight                                      #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday May 27th 2024 11:38:53 am                                                    #
# Modified   : Thursday June 20th 2024 02:10:45 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Dataset management module"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Type

from appinsight.utils.env import EnvManager
from appinsight.utils.file import IOService


# ------------------------------------------------------------------------------------------------ #
#                                         REPO                                                     #
# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Encapsulates persistence"""

    def __init__(
        self,
        efm_cls: Type[EnvManager] = EnvManager,
        io_cls: Type[IOService] = IOService,
    ) -> None:
        self.efm = efm_cls()
        self.io = io_cls()
        self._basedir = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def basedir(self) -> str:
        """Returns the base directory"""

    @abstractmethod
    def read(self, **kwargs) -> Any:
        """Reads data so designated."""

    @abstractmethod
    def write(self, directory: str, filename: str, data: Any) -> None:
        """Writes data to file."""

    @abstractmethod
    def get_filepath(self, **kwargs) -> str:
        """Returns the filepath for the current environment."""

    @abstractmethod
    def exists(self, **kwargs) -> str:
        """Returns the filepath for the current environment."""


# ------------------------------------------------------------------------------------------------ #
#                                      DATASET REPO                                                #
# ------------------------------------------------------------------------------------------------ #
class DatasetRepo(Repo):

    def __init__(self) -> None:
        super().__init__()
        self._basedir = "data"

    @property
    def basedir(self) -> str:
        """Returns the base directory"""
        return self._basedir

    def read(self, directory: str, filename: str) -> Any:
        """Reads data so designated."""
        filepath = self.get_filepath(directory=directory, filename=filename)
        return self.io.read(filepath=filepath)

    def write(self, directory: str, filename: str, data: Any) -> None:
        """Writes data to file."""
        filepath = self.get_filepath(directory=directory, filename=filename)
        self.io.write(filepath=filepath, data=data)

    def exists(self, directory: str, filename: str) -> str:
        """Returns the filepath for the current environment."""
        filepath = self.get_filepath(directory=directory, filename=filename)
        return os.path.exists(filepath)

    def get_filepath(self, directory: str, filename: str) -> str:
        """Returns the filepath for the current environment."""
        env = self.efm.get_environment()
        directory = os.path.join(self.basedir, env, directory)
        filename = f"{filename}"
        return os.path.join(directory, filename)
