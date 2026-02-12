import json
import re
import pandas as pd
from pathlib import Path


def parse_entities(entity_string):
    """
    解析实体字符串，将其转换为一个集合（忽略顺序）
    例如: '("CRC", Disease), ("lncTCF7", Gene Symbol)' -> {("CRC", "Disease"), ("lncTCF7", "Gene Symbol")}
    """
    if not entity_string or entity_string.strip() in ["null", "", "None"]:
        return set()
    
    # 使用正则表达式提取所有的 ("实体名", 实体类型) 模式
    pattern = r'\("([^"]+)",\s*([^)]+)\)'
    matches = re.findall(pattern, entity_string)
    
    # 转换为集合（自动去重和忽略顺序）
    return set((entity.strip(), entity_type.strip()) for entity, entity_type in matches)


def entities_are_equal(checked, check_myself):
    """
    比较两个实体字符串是否在内容上相等（忽略顺序）
    """
    set1 = parse_entities(checked)
    set2 = parse_entities(check_myself)
    return set1 == set2


def analyze_differences(json_file, output_dir=None):
    """
    分析 checked 和 check_myself 字段的差异
    """
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为 DataFrame
    raw_data = pd.DataFrame(data)
    
    print(f"总记录数: {len(raw_data)}")
    print("=" * 80)
    
    differences = []
    matches = []
    
    for i in range(len(raw_data)):
        checked = str(raw_data.loc[i, "checked"]) if "checked" in raw_data.columns else ""
        check_myself = str(raw_data.loc[i, "check_myself"]) if "check_myself" in raw_data.columns else ""
        
        # 保留原始记录的所有字段
        record = raw_data.loc[i].to_dict()
        
        # 使用集合比较，忽略顺序
        if not entities_are_equal(checked, check_myself):
            record['checked_set'] = parse_entities(checked)
            record['check_myself_set'] = parse_entities(check_myself)
            differences.append(record)
        else:
            # 保存没有差异的记录
            matches.append(record)
    
    print(f"\n实际有差异的记录数: {len(differences)}")
    print("=" * 80)
    
    # 打印每个差异
    for idx, diff in enumerate(differences, 1):
        print(f"\n【差异 #{idx}】 Index: {differences.index(diff)}")
        print(f"INPUT: {diff['INPUT'][:100]}..." if len(diff['INPUT']) > 100 else f"INPUT: {diff['INPUT']}")
        print(f"\nOUTPUT: {diff['OUTPUT']}")
        print(f"\ngt_r: {diff.get('gt_r', '')}")
        print(f"\nchecked: {diff['checked']}")
        print(f"\ncheck_myself: {diff['check_myself']}")
        
        # 显示差异分析
        checked_set = diff.get('checked_set', set())
        check_myself_set = diff.get('check_myself_set', set())
        
        only_in_checked = checked_set - check_myself_set
        only_in_check_myself = check_myself_set - checked_set
        
        print(f"\n【差异分析】")
        if only_in_checked:
            print(f"仅在 checked 中: {only_in_checked}")
        if only_in_check_myself:
            print(f"仅在 check_myself 中: {only_in_check_myself}")
        
        print("-" * 80)
    
    # 统计摘要
    print("\n" + "=" * 80)
    print(f"【统计摘要】")
    print(f"总记录数: {len(raw_data)}")
    print(f"完全一致的记录: {len(matches)}")
    print(f"有差异的记录: {len(differences)}")
    print(f"一致率: {len(matches) / len(raw_data) * 100:.2f}%")
    print("=" * 80)
    
    # 保存到 JSON 文件
    if output_dir is None:
        output_dir = Path(json_file).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存有差异的记录（移除set信息，只保留原始数据）
    differences_output = []
    for diff in differences:
        diff_copy = diff.copy()
        diff_copy.pop('checked_set', None)
        diff_copy.pop('check_myself_set', None)
        differences_output.append(diff_copy)
    
    diff_file = output_dir / "differences.json"
    with open(diff_file, 'w', encoding='utf-8') as f:
        json.dump(differences_output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 有差异的记录已保存到: {diff_file}")
    
    # 保存没有差异的记录
    match_file = output_dir / "matches.json"
    with open(match_file, 'w', encoding='utf-8') as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print(f"✅ 没有差异的记录已保存到: {match_file}")
    
    return differences, matches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='比较 checked 和 check_myself 字段的差异（忽略顺序）')
    parser.add_argument('-i', '--input', type=str, 
                        default='data/Entity_curation/gene_part_4_checked.json',
                        help='输入 JSON 文件路径')
    parser.add_argument('-o', '--output-dir', type=str,
                        default=None,
                        help='输出目录路径（默认与输入文件同目录）')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"错误: 文件 {args.input} 不存在！")
    else:
        analyze_differences(args.input, args.output_dir)
