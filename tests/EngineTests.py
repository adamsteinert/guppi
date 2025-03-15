import os, time
import unittest
from loguru import logger

from src.glados.Extensions.Commands.StoreMemoryCommand import StoreMemoryCommand
from src.glados.Extensions.Commands.commandtype import CommandType
from src.glados.Extensions.command_manager import command_manager
from src.glados.Extensions.Commands.NullCommand import NullCommand


#class EngineTests(unittest.TestCase):
