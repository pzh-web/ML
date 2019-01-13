#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/1/13 16:51
# @Author: peng yang
# @File  : strOperate.py


class StrOperate(object):

    @staticmethod
    def is_number(s):
        """
        判断是否是数字
        """
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False
