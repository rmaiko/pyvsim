"""
PyVSim v.1
Copyright 2013 Ricardo Entz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file lists the objects to be used in a normal execution of pyvsim. This
way, the user does not have to know the internal distribution of classes (which
may look somewhat arbitrary, e.g. Volume is in Primitives, not in Toolbox)

The down side is that one needs to register the classes here when the user
has to have access
"""
from Primitives import Assembly
from Primitives import Volume
from Primitives import RayBundle
from Toolbox import Mirror
from Toolbox import Dump
from Toolbox import Camera
from Toolbox import Laser
from System import save
from System import load
from System import plot
from Library import *
from Utils import readSTL
