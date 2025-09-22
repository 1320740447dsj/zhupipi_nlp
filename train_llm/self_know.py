import json

def process_jsonl_file(input_filename, output_filename):
    """
    处理jsonl文件，替换特定内容并转换格式。

    Args:
        input_filename (str): 输入的jsonl文件名。
        output_filename (str): 输出的jsonl文件名。
    """
    new_data = []

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():  # 跳过空行
                    data = json.loads(line)
                    
                    # 1. 替换 {{AUTHOR}}
                    # 2. 替换 {{NAME}}
                    new_response = data["response"].replace("{{AUTHOR}}", "二仙桥程序员").replace("{{NAME}}", "猪皮皮")

                    # 3. 按照新格式组织数据
                    new_entry = {
                        "conversations": [
                            {
                                "content": data["query"],
                                "role": "user"
                            },
                            {
                                "content": new_response,
                                "role": "assistant"
                            }
                        ]
                    }
                    new_data.append(new_entry)

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_filename}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_filename} 中的JSON格式不正确。")
        return

    # 将处理后的数据写入新的jsonl文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for item in new_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"文件处理完成！已将处理后的数据保存到 {output_filename}")

# --- 使用示例 ---
# 请将 'input.jsonl' 替换为您的原始文件名
# 'output.jsonl' 是生成的新文件名
input_file = 'self_cognition.jsonl'
output_file = 'self_cognitionv2.jsonl'
process_jsonl_file(input_file, output_file)