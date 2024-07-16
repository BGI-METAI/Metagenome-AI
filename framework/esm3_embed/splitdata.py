import pandas as pd

# 定义一个函数来处理序列数据 lable=1为 AMP 0为no-AMP
def process_sequence_data(file_path, lable =1):
    
    data_list = []  # 创建一个空列表来存储数据
    current_id = None
    current_sequence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):  # 检测是否为新的序列的开始
                if current_id is not None:  # 如果当前序列不为空，则保存到列表
                    data_list.append({
                        'ID': current_id,
                        'Sequence': '\n'.join(current_sequence)
                    })
                
                # 解析ID和名称
                parts = line[1:].split(maxsplit=1)  # 只分割一次
                current_id = parts[0] 
                current_sequence = []  # 重置序列列表
            else:
                current_sequence.append(line)  # 累加序列行

    # 保存最后一个序列
    if current_id is not None:
        data_list.append({
            'ID': current_id,
            'Sequence': '\n'.join(current_sequence)
        })

    # 将列表转换为DataFrame
    df = pd.DataFrame(data_list)
    # 添加一列 标签数据
    if lable == 1:
        df['lable'] = 1
    else:
        df['lable'] = 0
  
    return df
