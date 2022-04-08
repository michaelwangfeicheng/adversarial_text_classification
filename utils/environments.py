#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/3/25 11:26 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/25 11:26   wangfc      1.0         None
"""

import os

def set_working_dir(project_name='faq'):
    from pathlib import Path
    import sys
    if project_name:
        project_dir = Path.home().joinpath(project_name)
    else:
        script_path = os.path.abspath(__file__)
        project_dir = Path(script_path).parent.parent.absolute()
    os.chdir(project_dir)
    sys.path.extend([str(project_dir)])
    print(f"working_dir={os.getcwd()}")
    # print(f"sys.path={sys.path}")
    return project_dir
