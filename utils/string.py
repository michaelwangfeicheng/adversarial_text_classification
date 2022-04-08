#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/4 9:37 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/4 9:37   wangfc      1.0         None
"""
import re
from typing import Text


def rfind(string:Text,pattern:Text,begin:int=0,end:int=0):
    """
    str.rfind() 方法只支持字符向前的查找，而我们需要提供使用 正则的向前查找方法
    """
    start_position = end
    char = string[start_position]
    patter_complied = re.compile(pattern=pattern)
    default_position = begin - 1
    while start_position> default_position and not patter_complied.match(string=char) :
        start_position += -1
        char = string[start_position]
    return start_position

if __name__ == '__main__':
    title = '关于公司主体及相关债项信用等级移出信用评级观察名单的公告'
    content = '2018年8月31日，公司发布《湖北宜化化工股份有限公司2018年半年度报告》（以下\n简称“半年报”）。2018年上半年，公司主要产品市场价格回升，整体效益增加。同时，新\n疆宜化80.1%股权转让工作完成，新疆宜化不再纳入公司合并报表范围，并且新疆宜化股权\n转让过渡期损益由受让方承担，减少了公司当期损失。受上述因素影响，截至2018年6月\n末，公司总资产248.35亿元，较上年末下降23.70%；归属于上市公司股东的净资产8.15亿\n元，较上年末增长29.50%；2018年上半年，公司实现营业收入62.37亿元，同比下降0.92%；\n归属于上市公司股东的净利润2.38亿元，同比增长188.43%。\n鉴于公司实际控制人可能发生变更的事项已经终止，且已将新疆宜化剥离，上述事项对\n公司信用状况带来的不确定性影响已基本消除，联合评级通过对公司主体长期信用状况和\n“16宜化债”进行综合分析和评估，决定将公司主体及“16宜化债”债项信用等级移出信\n用评级观察名单，确定公司主体长期信用等级为“A-”，展望为“稳定”；“16宜化债”债项\n信用等级为“A-”。'
    title_content_joined_char = '。'
    entity_start_position =412
    entity_end_position = 417
    string = f"{title_content_joined_char}".join([title, content])
    pattern = r'[。|\n]'
    last_sentence_position = rfind(string=string, pattern=pattern, end=entity_start_position)
    context = string[last_sentence_position + 1:]
    new_entity_start_position = entity_start_position - last_sentence_position - 1
    new_entity_end_position = entity_end_position - last_sentence_position - 1
    # 验证新的 new_context中的entity位置
    entity_after_window = string[new_entity_start_position: new_entity_end_position].lower()
