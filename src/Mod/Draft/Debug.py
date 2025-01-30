# -*- coding: utf-8 -*-
# ***************************************************************************
# *   Copyright (c) 2024, 2025 aprospero <apro@posteo.net>        *
# *   Copyright (c) 2020 FreeCAD Developers                                 *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU Lesser General Public License (LGPL)    *
# *   as published by the Free Software Foundation; either version 2 of     *
# *   the License, or (at your option) any later version.                   *
# *   for detail see the LICENCE text file.                                 *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU Library General Public License for more details.                  *
# *                                                                         *
# *   You should have received a copy of the GNU Library General Public     *
# *   License along with this program; if not, write to the Free Software   *
# *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  *
# *   USA                                                                   *
# *                                                                         *
# ***************************************************************************
"""Provide the Debug public programming interface.

The Debug module offers tools.
These functions can be used.
"""
## \addtogroup Debug
#  @{


def prepare():
    """prepare a pydev debug session.

    Parameters
    ----------
 
    Returns
    -------
    """
    import sys
    sys.path.append("/home/tidy/.local/opt/liclipse/plugins/org.python.pydev.core_12.1.0.202405280850/pysrc/")
    import pydevd

def start():
    """start a pydev debug session.

    Parameters
    ----------
 
    Returns
    -------
    """
    import pydevd
    pydevd.settrace()
    



## @}
