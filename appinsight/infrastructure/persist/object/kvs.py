#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : AppInsight                                                                          #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.3                                                                              #
# Filename   : /appinsight/infrastructure/persist/object/kvs.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john@variancexplained.com                                                           #
# URL        : https://github.com/variancexplained/appinsight                                      #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 30th 2024 10:27:18 pm                                                   #
# Modified   : Monday July 1st 2024 11:58:13 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Key Value Store Module"""
import logging
import os
import shelve
from typing import Any

from appinsight.infrastructure.persist.file.io import IOService
from appinsight.infrastructure.utils.env import EnvManager


class KVS:
    """Manages access to key/value (department) store for the project.

    The KVS is organized by department and name. This class expects that the
    environment configuration file has specified the location of the department
    and name under the 'kvs' key in the configuration files.

    Args:
        department (str): The department can be 'cache', 'repo', etc.
        name (str): The name of the KVS within the department.
        env_mgr_cls (type[EnvManager], optional): Environment manager class.
        io_cls (type[IOService], optional): IO service class.

    Examples:
        >>> kvs = KVS(department='cache', name='example')
        >>> kvs.create('key1', 'value1')
        >>> kvs.read('key1')
        >>> kvs.delete('key1')
    """

    def __init__(
        self,
        department: str,
        name: str,
        env_mgr_cls: type[EnvManager] = EnvManager,
        io_cls: type[IOService] = IOService,
    ) -> None:
        self._department = department
        self._name = name
        self._env_mgr = env_mgr_cls()
        self._io = io_cls()

        self._kvs_file = self._get_filepath(department=department, name=name)
        self._validate_kvs()

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def __len__(self) -> int:
        """Returns the number of keys in the KVS."""
        with shelve.open(self._kvs_file) as kvs:
            return len(kvs)

    def __iter__(self):
        """Returns an iterator for iterating through keys in the KVS."""
        self._kvs = shelve.open(self._kvs_file)
        self._iter_keys = iter(self._kvs.keys())
        return self

    def __next__(self) -> Any:
        """Returns the next key in the iteration."""
        try:
            key = next(self._iter_keys)
            return key
        except StopIteration:
            self._kvs.close()
            raise

    def create(self, key: str, value: Any) -> None:
        """Persists the object in the KVS.

        Args:
            key (str): Key for the entry.
            value (Any): Value to be stored.

        Raises:
            FileExistsError: If the key already exists.
        """
        if self.exists(key):
            msg = f"Cannot create {key} as it already exists."
            self._logger.exception(msg)
            raise FileExistsError(msg)

        with shelve.open(self._kvs_file) as kvs:
            kvs[key] = value
        self._logger.debug(f"Added {key} to KVS.")

    def read(self, key: str) -> Any:
        """Reads an object from the KVS.

        Args:
            key (str): Key for the entry.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            with shelve.open(self._kvs_file) as kvs:
                return kvs[key]
        except KeyError as ke:
            msg = f"No key {key} found.\n{ke}"
            self._logger.exception(msg)
            raise

    def delete(self, key: str) -> None:
        """Deletes an object from the KVS.

        Args:
            key (str): Key for the entry.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            with shelve.open(self._kvs_file) as kvs:
                del kvs[key]
        except KeyError as ke:
            msg = f"No key {key} found.\n{ke}"
            self._logger.warning(msg)

    def exists(self, key: str) -> bool:
        """Checks if a key exists in the KVS.

        Args:
            key (str): Key for the entry.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        with shelve.open(self._kvs_file) as kvs:
            return key in kvs

    def reset(self) -> None:
        """Resets the KVS by clearing all entries.

        Raises:
            Exception: For any issues during the reset operation.
        """
        confirmation = input(
            f"Are you sure you want to reset the KVS '{self._name}' in the department '{self._department}'? This action cannot be undone. (yes/no): "
        )
        if confirmation.lower() != "yes":
            print("Reset operation aborted.")
            return

        try:
            with shelve.open(self._kvs_file) as kvs:
                for key in list(kvs.keys()):
                    del kvs[key]
            self._logger.info("KVS has been reset.")
        except Exception as e:
            msg = f"Exception occurred while resetting {self._kvs_file}.\n{e}"
            self._logger.exception(msg)
            raise

    def _get_filepath(self, department: str, name: str) -> str:
        """Returns the KVS file path for the department and name from config.

        Args:
            department (str): Department name.
            name (str): KVS name.

        Returns:
            str: The file path of the KVS.

        Raises:
            ValueError: If the KVS configuration is invalid.
        """
        env = self._env_mgr.get_environment()
        config_filepath = os.path.join("config", f"{env.lower()}.yml")
        config = IOService.read(filepath=config_filepath)
        try:
            return config["kvs"][department][name]
        except KeyError as ke:
            msg = f"No key value store exists for {department}.{name}.\n{ke}"
            logging.exception(msg)
            raise ValueError()
        except Exception as e:
            msg = f"Unknown error occurred while accessing the KVS configuration.\n{e}"
            self._logger.exception(msg)
            raise

    def _validate_kvs(self) -> None:
        """Validates the KVS is accessible.

        Ensures the directory exists in case this process creates the kvs file.

        Raises:
            FileNotFoundError: If the KVS file does not exist.
            Exception: For other issues accessing the KVS file.
        """
        os.makedirs(os.path.dirname(self._kvs_file), exist_ok=True)
        try:
            with shelve.open(self._kvs_file):
                return
        except FileNotFoundError as fe:
            self._logger.exception(f"KVS file {self._kvs_file} does not exist.\n{fe}")
            raise
        except Exception as e:
            self._logger.exception(
                f"Exception occurred while reading {self._kvs_file}.\n{e}"
            )
            raise
