import pandas as pd
import json
from pathlib import Path
from settings.logger_manager import get_logger

logger = get_logger(__name__)

# 表格切片类
class TableChunk:
    """表格切片类：封装表格切片逻辑"""
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.metaflag = []
        
        # 根据文件扩展名选择读取方式
        file_ext = Path(input_file).suffix.lower()
        if file_ext == '.csv':
            # CSV文件：读取为单Sheet结构，尝试多种编码
            df = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    logger.info(f"CSV文件读取成功，使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                # 如果所有编码都失败，使用默认编码并忽略错误
                df = pd.read_csv(input_file, encoding='utf-8', encoding_errors='ignore')
                logger.warning("CSV文件使用默认编码读取，可能存在乱码")
            self.df = {'_csv_default': df}
        else:
            # Excel文件：读取所有工作表
            self.df = pd.read_excel(input_file, sheet_name=None)

    # 设置元数据标志
    def set_metaflag(self, metaflag: list):
        """设置元数据标志"""
        """
        metaflag: 元数据标志字典，示例如下：
        --- excel文档的元数据标志 - --
        [
            "source",  # 左边知识库文档中勾选的某个文档名称
            "sheet",  # 下拉框选择的sheet表
            "序号",  # 网页上设置的元素据标签
            "故障分类",  # 网页上设置的元素据标签
            "严重程度",  # 网页上设置的元素据标签
            "适用机型"  # 网页上设置的元素据标签
        ]

        ---csv文档的元数据标志 - --
        [
            "source",  # 左边知识库文档中勾选的某个文档名称
            "序号",  # 网页上设置的元素据标签
            "故障分类",  # 网页上设置的元素据标签
            "严重程度",  # 网页上设置的元素据标签
            "适用机型"  # 网页上设置的元素据标签
        ]
        """
        self.metaflag = metaflag

    # 表格内容切片
    def chunk_table(self):
        chunk_list = []

        for sheet_name, sheet_df in self.df.items():
            # 过滤空Sheet表
            if sheet_df.empty or len(sheet_df.columns) == 0:
                logger.info(f"跳过空Sheet表: {sheet_name}")
                continue
                
            # 剩余列作为正文列
            content_cols = [c for c in sheet_df.columns if str(c).strip() not in self.metaflag]

            logger.info(f"\n[{sheet_name}]")
            logger.info(f"  总列数: {len(sheet_df.columns)}")
            logger.info(f"  标签列: {self.metaflag}")
            logger.info(f"  正文列: {content_cols}")
            logger.info(f"  总行数: {len(sheet_df)}")

            # 如果所有列都被选为元数据标签，则使用所有列作为正文列
            if not content_cols:
                logger.info(f"  所有列都被选为元数据标签，使用所有列作为正文")
                content_cols = list(sheet_df.columns)

            for i, (idx, row) in enumerate(sheet_df.iterrows()):
                # 拼接正文
                parts = []
                for col in content_cols:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        parts.append(f"【{col}】{val}")
                if not parts:
                    continue

                chunk_text = "\n".join(parts)

                # 构建 metadata
                metadata = {
                    "source": self.input_file,
                }
                # CSV文件不添加sheet元数据
                if "sheet" in self.metaflag and sheet_name != '_csv_default':
                    metadata["sheet"] = sheet_name

                for meta in self.metaflag:
                    if meta in content_cols:
                        continue
                    elif meta == "source" or meta == "sheet":
                        continue

                    val = row[meta]
                    if pd.notna(val) and str(val).strip():
                        metadata[meta] = str(val).strip()

                chunk_list.append({
                    "chunk_text": chunk_text,
                    "metadata": metadata
                })

        logger.info(f"共生成 {len(chunk_list)} 个 chunk")
        if chunk_list:
            logger.info("--- 第1个 chunk ---")
            logger.info(chunk_list[0]["chunk_text"])
            logger.info("--- 第1个 元数据 ---")
            logger.info(json.dumps(chunk_list[0]["metadata"], ensure_ascii=False, indent=2))

        return chunk_list


# if __name__ == "__main__":
#     table_chunk = TableChunk("D:\\知识库素材\\手机相机故障诊断_知识库.xlsx")
#     table_chunk.set_metaflag(["source", "sheet", "序号", "故障分类", "严重程度", "适用机型"])
#     chunk_list = table_chunk.chunk()