# -*- coding: utf-8 -*-
"""构建海外高校排名知识库 overseas_school_rankings.json

数据源：
  - QS World University Rankings 2025
  - U.S. News Best Global Universities 2024-2025

输出：data/knowledge/overseas_school_rankings.json
层级映射规则：任一榜单 TOP50 -> 985；TOP51-100 -> 211；TOP101-150 -> 双一流。
两榜单取更高层级（更优名次）。校名采用 英文(中文) 双语。
"""
import io
import sys
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent.parent / "knowledge" / "overseas_school_rankings.json"

# ── QS World University Rankings 2025 TOP150 ──────────────────────────────
# 格式: (rank, name_en, name_cn, country)
QS_2025 = [
    (1, "Massachusetts Institute of Technology", "麻省理工学院", "美国"),
    (2, "Imperial College London", "帝国理工学院", "英国"),
    (3, "University of Oxford", "牛津大学", "英国"),
    (4, "Harvard University", "哈佛大学", "美国"),
    (5, "University of Cambridge", "剑桥大学", "英国"),
    (6, "Stanford University", "斯坦福大学", "美国"),
    (7, "ETH Zurich", "苏黎世联邦理工学院", "瑞士"),
    (8, "National University of Singapore", "新加坡国立大学", "新加坡"),
    (9, "University College London", "伦敦大学学院", "英国"),
    (10, "California Institute of Technology", "加州理工学院", "美国"),
    (11, "University of Pennsylvania", "宾夕法尼亚大学", "美国"),
    (12, "University of California, Berkeley", "加州大学伯克利分校", "美国"),
    (13, "The University of Melbourne", "墨尔本大学", "澳大利亚"),
    (14, "Peking University", "北京大学", "中国"),
    (15, "Nanyang Technological University", "南洋理工大学", "新加坡"),
    (16, "Cornell University", "康奈尔大学", "美国"),
    (17, "The University of Hong Kong", "香港大学", "中国香港"),
    (18, "The University of Sydney", "悉尼大学", "澳大利亚"),
    (19, "University of New South Wales", "新南威尔士大学", "澳大利亚"),
    (20, "Tsinghua University", "清华大学", "中国"),
    (21, "University of Chicago", "芝加哥大学", "美国"),
    (22, "Princeton University", "普林斯顿大学", "美国"),
    (23, "Yale University", "耶鲁大学", "美国"),
    (24, "University of Toronto", "多伦多大学", "加拿大"),
    (25, "The University of Edinburgh", "爱丁堡大学", "英国"),
    (26, "Ecole Polytechnique Federale de Lausanne", "洛桑联邦理工学院", "瑞士"),
    (27, "The University of Manchester", "曼彻斯特大学", "英国"),
    (28, "The Hong Kong University of Science and Technology", "香港科技大学", "中国香港"),
    (29, "Australian National University", "澳大利亚国立大学", "澳大利亚"),
    (30, "McGill University", "麦吉尔大学", "加拿大"),
    (31, "Technical University of Munich", "慕尼黑工业大学", "德国"),
    (32, "Johns Hopkins University", "约翰斯·霍普金斯大学", "美国"),
    (33, "Seoul National University", "首尔国立大学", "韩国"),
    (34, "The University of Tokyo", "东京大学", "日本"),
    (35, "The University of British Columbia", "英属哥伦比亚大学", "加拿大"),
    (36, "Fudan University", "复旦大学", "中国"),
    (37, "Zhejiang University", "浙江大学", "中国"),
    (38, "King's College London", "伦敦国王学院", "英国"),
    (39, "University of California, Los Angeles", "加州大学洛杉矶分校", "美国"),
    (40, "Shanghai Jiao Tong University", "上海交通大学", "中国"),
    (41, "Universite PSL", "巴黎文理研究大学", "法国"),
    (42, "Kyoto University", "京都大学", "日本"),
    (43, "Korea Advanced Institute of Science and Technology", "韩国科学技术院", "韩国"),
    (44, "University of Michigan-Ann Arbor", "密歇根大学安娜堡分校", "美国"),
    (45, "The Chinese University of Hong Kong", "香港中文大学", "中国香港"),
    (46, "New York University", "纽约大学", "美国"),
    (47, "The Hong Kong Polytechnic University", "香港理工大学", "中国香港"),
    (48, "Sorbonne University", "索邦大学", "法国"),
    (49, "Universiti Malaya", "马来亚大学", "马来西亚"),
    (50, "Institut Polytechnique de Paris", "巴黎综合理工学院", "法国"),
    (51, "University of Bristol", "布里斯托大学", "英国"),
    (52, "University of Amsterdam", "阿姆斯特丹大学", "荷兰"),
    (53, "Nanjing University", "南京大学", "中国"),
    (54, "Delft University of Technology", "代尔夫特理工大学", "荷兰"),
    (55, "Monash University", "莫纳什大学", "澳大利亚"),
    (56, "The University of Texas at Austin", "德克萨斯大学奥斯汀分校", "美国"),
    (57, "Duke University", "杜克大学", "美国"),
    (58, "University of California, San Diego", "加州大学圣地亚哥分校", "美国"),
    (59, "Northwestern University", "西北大学（美国）", "美国"),
    (60, "The University of Queensland", "昆士兰大学", "澳大利亚"),
    (61, "Carnegie Mellon University", "卡内基梅隆大学", "美国"),
    (62, "City University of Hong Kong", "香港城市大学", "中国香港"),
    (63, "University of Washington", "华盛顿大学", "美国"),
    (64, "Ludwig-Maximilians-Universitat Munchen", "慕尼黑大学", "德国"),
    (65, "University of Illinois at Urbana-Champaign", "伊利诺伊大学厄巴纳-香槟分校", "美国"),
    (66, "The London School of Economics and Political Science", "伦敦政治经济学院", "英国"),
    (67, "The University of Warwick", "华威大学", "英国"),
    (68, "The University of Auckland", "奥克兰大学", "新西兰"),
    (69, "National Taiwan University", "台湾大学", "中国台湾"),
    (70, "Universidad de Buenos Aires", "布宜诺斯艾利斯大学", "阿根廷"),
    (71, "Brown University", "布朗大学", "美国"),
    (72, "KU Leuven", "鲁汶大学", "比利时"),
    (73, "University of Bristol", "布里斯托大学", "英国"),
    (74, "Yonsei University", "延世大学", "韩国"),
    (75, "University of Birmingham", "伯明翰大学", "英国"),
    (76, "Korea University", "高丽大学", "韩国"),
    (77, "University of Southampton", "南安普顿大学", "英国"),
    (78, "Lund University", "隆德大学", "瑞典"),
    (79, "University of Leeds", "利兹大学", "英国"),
    (80, "The University of Western Australia", "西澳大学", "澳大利亚"),
    (81, "Trinity College Dublin", "都柏林圣三一学院", "爱尔兰"),
    (82, "Universidad Nacional Autonoma de Mexico", "墨西哥国立自治大学", "墨西哥"),
    (83, "Pohang University of Science and Technology", "浦项科技大学", "韩国"),
    (84, "The Pennsylvania State University", "宾夕法尼亚州立大学", "美国"),
    (85, "Durham University", "杜伦大学", "英国"),
    (86, "Tokyo Institute of Technology", "东京工业大学", "日本"),
    (87, "University of Zurich", "苏黎世大学", "瑞士"),
    (88, "University of St Andrews", "圣安德鲁斯大学", "英国"),
    (89, "Boston University", "波士顿大学", "美国"),
    (90, "Osaka University", "大阪大学", "日本"),
    (91, "Sungkyunkwan University", "成均馆大学", "韩国"),
    (92, "University of Copenhagen", "哥本哈根大学", "丹麦"),
    (93, "University of Technology Sydney", "悉尼科技大学", "澳大利亚"),
    (94, "Universidade de Sao Paulo", "圣保罗大学", "巴西"),
    (95, "Tohoku University", "东北大学（日本）", "日本"),
    (96, "University of Wisconsin-Madison", "威斯康星大学麦迪逊分校", "美国"),
    (97, "Universiti Putra Malaysia", "马来西亚博特拉大学", "马来西亚"),
    (98, "Georgia Institute of Technology", "佐治亚理工学院", "美国"),
    (99, "Universiti Kebangsaan Malaysia", "马来西亚国民大学", "马来西亚"),
    (100, "KTH Royal Institute of Technology", "瑞典皇家理工学院", "瑞典"),
    (101, "Universiti Sains Malaysia", "马来西亚理科大学", "马来西亚"),
    (102, "University of Glasgow", "格拉斯哥大学", "英国"),
    (103, "Universite de Montreal", "蒙特利尔大学", "加拿大"),
    (104, "University of Alberta", "阿尔伯塔大学", "加拿大"),
    (105, "University of Nottingham", "诺丁汉大学", "英国"),
    (106, "Universiti Teknologi Malaysia", "马来西亚理工大学", "马来西亚"),
    (107, "Pontificia Universidad Catolica de Chile", "智利天主教大学", "智利"),
    (108, "RWTH Aachen University", "亚琛工业大学", "德国"),
    (109, "Lomonosov Moscow State University", "莫斯科国立罗蒙诺索夫大学", "俄罗斯"),
    (110, "Freie Universitaet Berlin", "柏林自由大学", "德国"),
    (111, "King Abdulaziz University", "阿卜杜勒阿齐兹国王大学", "沙特阿拉伯"),
    (112, "Western University", "西安大略大学", "加拿大"),
    (113, "Wuhan University", "武汉大学", "中国"),
    (114, "University of Adelaide", "阿德莱德大学", "澳大利亚"),
    (115, "Universidad de Chile", "智利大学", "智利"),
    (116, "Stockholm University", "斯德哥尔摩大学", "瑞典"),
    (117, "Karlsruhe Institute of Technology", "卡尔斯鲁厄理工学院", "德国"),
    (118, "Purdue University", "普渡大学", "美国"),
    (119, "University of Groningen", "格罗宁根大学", "荷兰"),
    (120, "The Hong Kong Baptist University", "香港浸会大学", "中国香港"),
    (121, "University of Bern", "伯尔尼大学", "瑞士"),
    (122, "Lancaster University", "兰卡斯特大学", "英国"),
    (123, "University of Cape Town", "开普敦大学", "南非"),
    (124, "Eindhoven University of Technology", "埃因霍温理工大学", "荷兰"),
    (125, "Harbin Institute of Technology", "哈尔滨工业大学", "中国"),
    (126, "Aalto University", "阿尔托大学", "芬兰"),
    (127, "Universidad Autonoma de Madrid", "马德里自治大学", "西班牙"),
    (128, "University of Maryland, College Park", "马里兰大学帕克分校", "美国"),
    (129, "University of Helsinki", "赫尔辛基大学", "芬兰"),
    (130, "Tongji University", "同济大学", "中国"),
    (131, "Michigan State University", "密歇根州立大学", "美国"),
    (132, "Uppsala University", "乌普萨拉大学", "瑞典"),
    (133, "University of Oslo", "奥斯陆大学", "挪威"),
    (134, "University of Vienna", "维也纳大学", "奥地利"),
    (135, "University of Sheffield", "谢菲尔德大学", "英国"),
    (136, "University of Science and Technology of China", "中国科学技术大学", "中国"),
    (137, "Cardiff University", "卡迪夫大学", "英国"),
    (138, "Humboldt-Universitat zu Berlin", "柏林洪堡大学", "德国"),
    (139, "University of Bath", "巴斯大学", "英国"),
    (140, "Universita di Bologna", "博洛尼亚大学", "意大利"),
    (141, "Ghent University", "根特大学", "比利时"),
    (142, "University of Geneva", "日内瓦大学", "瑞士"),
    (143, "Universita degli Studi di Padova", "帕多瓦大学", "意大利"),
    (144, "University of Southern California", "南加州大学", "美国"),
    (145, "Sapienza University of Rome", "罗马第一大学", "意大利"),
    (146, "Indian Institute of Technology Bombay", "印度理工学院孟买分校", "印度"),
    (147, "Queen Mary University of London", "伦敦玛丽女王大学", "英国"),
    (148, "University of Basel", "巴塞尔大学", "瑞士"),
    (149, "Sun Yat-sen University", "中山大学", "中国"),
    (150, "Newcastle University", "纽卡斯尔大学", "英国"),
]

# ── U.S. News Best Global Universities 2024-2025 TOP150 ───────────────────
USNEWS_2025 = [
    (1, "Harvard University", "哈佛大学", "美国"),
    (2, "Massachusetts Institute of Technology", "麻省理工学院", "美国"),
    (3, "Stanford University", "斯坦福大学", "美国"),
    (4, "University of Oxford", "牛津大学", "英国"),
    (5, "University of California, Berkeley", "加州大学伯克利分校", "美国"),
    (6, "University of Cambridge", "剑桥大学", "英国"),
    (7, "University of Washington", "华盛顿大学", "美国"),
    (8, "Princeton University", "普林斯顿大学", "美国"),
    (9, "Yale University", "耶鲁大学", "美国"),
    (10, "California Institute of Technology", "加州理工学院", "美国"),
    (11, "Columbia University", "哥伦比亚大学", "美国"),
    (12, "University of Pennsylvania", "宾夕法尼亚大学", "美国"),
    (13, "University of California, Los Angeles", "加州大学洛杉矶分校", "美国"),
    (14, "Johns Hopkins University", "约翰斯·霍普金斯大学", "美国"),
    (15, "University of California, San Francisco", "加州大学旧金山分校", "美国"),
    (16, "Tsinghua University", "清华大学", "中国"),
    (17, "University of Toronto", "多伦多大学", "加拿大"),
    (18, "Imperial College London", "帝国理工学院", "英国"),
    (19, "University of Michigan-Ann Arbor", "密歇根大学安娜堡分校", "美国"),
    (20, "University College London", "伦敦大学学院", "英国"),
    (21, "Cornell University", "康奈尔大学", "美国"),
    (22, "University of Chicago", "芝加哥大学", "美国"),
    (23, "ETH Zurich", "苏黎世联邦理工学院", "瑞士"),
    (24, "Peking University", "北京大学", "中国"),
    (25, "Duke University", "杜克大学", "美国"),
    (26, "Northwestern University", "西北大学（美国）", "美国"),
    (27, "University of California, San Diego", "加州大学圣地亚哥分校", "美国"),
    (28, "National University of Singapore", "新加坡国立大学", "新加坡"),
    (29, "New York University", "纽约大学", "美国"),
    (30, "University of Melbourne", "墨尔本大学", "澳大利亚"),
    (31, "University of Edinburgh", "爱丁堡大学", "英国"),
    (32, "Zhejiang University", "浙江大学", "中国"),
    (33, "Shanghai Jiao Tong University", "上海交通大学", "中国"),
    (34, "University of British Columbia", "英属哥伦比亚大学", "加拿大"),
    (35, "Washington University in St. Louis", "圣路易斯华盛顿大学", "美国"),
    (36, "Sorbonne University", "索邦大学", "法国"),
    (37, "University of Wisconsin-Madison", "威斯康星大学麦迪逊分校", "美国"),
    (38, "University of Sydney", "悉尼大学", "澳大利亚"),
    (39, "Chinese University of Hong Kong", "香港中文大学", "中国香港"),
    (40, "University of Texas at Austin", "德克萨斯大学奥斯汀分校", "美国"),
    (41, "University of Hong Kong", "香港大学", "中国香港"),
    (42, "University of Munich", "慕尼黑大学", "德国"),
    (43, "University of Queensland", "昆士兰大学", "澳大利亚"),
    (44, "University of North Carolina at Chapel Hill", "北卡罗来纳大学教堂山分校", "美国"),
    (45, "Karolinska Institute", "卡罗林斯卡学院", "瑞典"),
    (46, "Fudan University", "复旦大学", "中国"),
    (47, "University of Minnesota Twin Cities", "明尼苏达大学双城分校", "美国"),
    (48, "Technical University of Munich", "慕尼黑工业大学", "德国"),
    (49, "University of Manchester", "曼彻斯特大学", "英国"),
    (50, "University of Pittsburgh", "匹兹堡大学", "美国"),
    (51, "Monash University", "莫纳什大学", "澳大利亚"),
    (52, "Heidelberg University", "海德堡大学", "德国"),
    (53, "Kyoto University", "京都大学", "日本"),
    (54, "University of Copenhagen", "哥本哈根大学", "丹麦"),
    (55, "King's College London", "伦敦国王学院", "英国"),
    (56, "University of Science and Technology of China", "中国科学技术大学", "中国"),
    (57, "University of Maryland, College Park", "马里兰大学帕克分校", "美国"),
    (58, "University of New South Wales", "新南威尔士大学", "澳大利亚"),
    (59, "Nanyang Technological University", "南洋理工大学", "新加坡"),
    (60, "University of Tokyo", "东京大学", "日本"),
    (61, "McGill University", "麦吉尔大学", "加拿大"),
    (62, "University of Southern California", "南加州大学", "美国"),
    (63, "Boston University", "波士顿大学", "美国"),
    (64, "Australian National University", "澳大利亚国立大学", "澳大利亚"),
    (65, "Sun Yat-sen University", "中山大学", "中国"),
    (66, "Huazhong University of Science and Technology", "华中科技大学", "中国"),
    (67, "Purdue University", "普渡大学", "美国"),
    (68, "University of California, Davis", "加州大学戴维斯分校", "美国"),
    (69, "Ohio State University", "俄亥俄州立大学", "美国"),
    (70, "Xi'an Jiaotong University", "西安交通大学", "中国"),
    (71, "Wuhan University", "武汉大学", "中国"),
    (72, "Utrecht University", "乌得勒支大学", "荷兰"),
    (73, "University of Amsterdam", "阿姆斯特丹大学", "荷兰"),
    (74, "KU Leuven", "鲁汶大学", "比利时"),
    (75, "University of Bristol", "布里斯托大学", "英国"),
    (76, "Emory University", "埃默里大学", "美国"),
    (77, "University of Zurich", "苏黎世大学", "瑞士"),
    (78, "Sichuan University", "四川大学", "中国"),
    (79, "Tongji University", "同济大学", "中国"),
    (80, "University of Helsinki", "赫尔辛基大学", "芬兰"),
    (81, "Lund University", "隆德大学", "瑞典"),
    (82, "Carnegie Mellon University", "卡内基梅隆大学", "美国"),
    (83, "Nanjing University", "南京大学", "中国"),
    (84, "University of Oslo", "奥斯陆大学", "挪威"),
    (85, "Erasmus University Rotterdam", "鹿特丹伊拉斯姆斯大学", "荷兰"),
    (86, "Aarhus University", "奥胡斯大学", "丹麦"),
    (87, "University of Colorado Boulder", "科罗拉多大学博尔德分校", "美国"),
    (88, "Ghent University", "根特大学", "比利时"),
    (89, "Michigan State University", "密歇根州立大学", "美国"),
    (90, "University of Geneva", "日内瓦大学", "瑞士"),
    (91, "Seoul National University", "首尔国立大学", "韩国"),
    (92, "University of Adelaide", "阿德莱德大学", "澳大利亚"),
    (93, "Vanderbilt University", "范德堡大学", "美国"),
    (94, "Pennsylvania State University", "宾夕法尼亚州立大学", "美国"),
    (95, "University of California, Santa Barbara", "加州大学圣塔芭芭拉分校", "美国"),
    (96, "Leiden University", "莱顿大学", "荷兰"),
    (97, "University of Groningen", "格罗宁根大学", "荷兰"),
    (98, "University of Western Australia", "西澳大学", "澳大利亚"),
    (99, "Uppsala University", "乌普萨拉大学", "瑞典"),
    (100, "Southern University of Science and Technology", "南方科技大学", "中国"),
    (101, "University of Bonn", "波恩大学", "德国"),
    (102, "Technical University of Denmark", "丹麦技术大学", "丹麦"),
    (103, "University of Nottingham", "诺丁汉大学", "英国"),
    (104, "Hong Kong University of Science and Technology", "香港科技大学", "中国香港"),
    (105, "University of Glasgow", "格拉斯哥大学", "英国"),
    (106, "RWTH Aachen University", "亚琛工业大学", "德国"),
    (107, "Beijing Normal University", "北京师范大学", "中国"),
    (108, "Tianjin University", "天津大学", "中国"),
    (109, "University of Birmingham", "伯明翰大学", "英国"),
    (110, "University of Bern", "伯尔尼大学", "瑞士"),
    (111, "University of Florida", "佛罗里达大学", "美国"),
    (112, "Karlsruhe Institute of Technology", "卡尔斯鲁厄理工学院", "德国"),
    (113, "Delft University of Technology", "代尔夫特理工大学", "荷兰"),
    (114, "Rice University", "莱斯大学", "美国"),
    (115, "Stockholm University", "斯德哥尔摩大学", "瑞典"),
    (116, "Southeast University", "东南大学", "中国"),
    (117, "Arizona State University", "亚利桑那州立大学", "美国"),
    (118, "Harbin Institute of Technology", "哈尔滨工业大学", "中国"),
    (119, "Radboud University Nijmegen", "拉德堡德大学", "荷兰"),
    (120, "University of Warwick", "华威大学", "英国"),
    (121, "Tel Aviv University", "特拉维夫大学", "以色列"),
    (122, "University of Vienna", "维也纳大学", "奥地利"),
    (123, "Beihang University", "北京航空航天大学", "中国"),
    (124, "Hebrew University of Jerusalem", "希伯来大学", "以色列"),
    (125, "Eindhoven University of Technology", "埃因霍温理工大学", "荷兰"),
    (126, "Shandong University", "山东大学", "中国"),
    (127, "University of Sheffield", "谢菲尔德大学", "英国"),
    (128, "University of Leeds", "利兹大学", "英国"),
    (129, "University of Barcelona", "巴塞罗那大学", "西班牙"),
    (130, "Universite Paris-Saclay", "巴黎萨克雷大学", "法国"),
    (131, "Dalian University of Technology", "大连理工大学", "中国"),
    (132, "University of Southampton", "南安普顿大学", "英国"),
    (133, "Jilin University", "吉林大学", "中国"),
    (134, "Xiamen University", "厦门大学", "中国"),
    (135, "University of Goettingen", "哥廷根大学", "德国"),
    (136, "Aalto University", "阿尔托大学", "芬兰"),
    (137, "University of Cape Town", "开普敦大学", "南非"),
    (138, "Lomonosov Moscow State University", "莫斯科国立大学", "俄罗斯"),
    (139, "Central South University", "中南大学", "中国"),
    (140, "Hunan University", "湖南大学", "中国"),
    (141, "University of Freiburg", "弗莱堡大学", "德国"),
    (142, "City University of Hong Kong", "香港城市大学", "中国香港"),
    (143, "University of Notre Dame", "圣母大学", "美国"),
    (144, "Trinity College Dublin", "都柏林圣三一学院", "爱尔兰"),
    (145, "University of Tubingen", "图宾根大学", "德国"),
    (146, "University of Rochester", "罗切斯特大学", "美国"),
    (147, "Tohoku University", "东北大学（日本）", "日本"),
    (148, "Cardiff University", "卡迪夫大学", "英国"),
    (149, "Indian Institute of Science", "印度科学理工学院", "印度"),
    (150, "Newcastle University", "纽卡斯尔大学", "英国"),
]


def rank_to_tier(rank: int) -> str:
    if rank <= 50:
        return "985"
    if rank <= 100:
        return "211"
    return "双一流"


def main():
    # 合并去重：以英文校名为 key，记录两个榜单的名次
    merged = {}

    def add(lst, key):
        for rank, en, cn, country in lst:
            norm = en.strip()
            if norm not in merged:
                merged[norm] = {
                    "name_en": en,
                    "name_cn": cn,
                    "display": f"{en}({cn})",
                    "country": country,
                    "qs_2025_rank": None,
                    "usnews_2025_rank": None,
                }
            merged[norm][key] = rank

    add(QS_2025, "qs_2025_rank")
    add(USNEWS_2025, "usnews_2025_rank")

    # 计算等价层级：两榜取更优名次映射
    for v in merged.values():
        ranks = [r for r in (v["qs_2025_rank"], v["usnews_2025_rank"]) if r]
        best = min(ranks)
        v["best_rank"] = best
        v["equiv_tier"] = rank_to_tier(best)

    universities = sorted(merged.values(), key=lambda x: x["best_rank"])

    tier_count = {"985": 0, "211": 0, "双一流": 0}
    for v in universities:
        tier_count[v["equiv_tier"]] += 1

    out = {
        "_metadata": {
            "description": "海外高校权威排名知识库 - QS 2025 与 U.S. News 2024-2025 前150名，用于HR智能匹配中对留学经历候选人的院校质量评估",
            "tier_mapping_rule": {
                "top_1_50": "985（顶尖海外名校）",
                "top_51_100": "211",
                "top_101_150": "双一流",
                "note": "两个榜单取更优名次（更小rank）所对应的层级",
            },
            "naming_convention": "校名采用 英文(中文) 双语，display字段可直接用于简历文本",
            "source": [
                "QS World University Rankings 2025",
                "U.S. News Best Global Universities 2024-2025",
            ],
            "total_universities": len(universities),
            "tier_distribution": tier_count,
        },
        "universities": universities,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已写入 {OUT}")
    print(f"[OK] 合并去重后院校总数: {len(universities)}")
    print(f"[OK] 等价层级分布: {tier_count}")
    # 抽样打印
    for v in universities[:5]:
        print(f"  {v['best_rank']:>3} {v['equiv_tier']} {v['display']}")


if __name__ == "__main__":
    main()
