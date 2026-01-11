# OCR 多模态数据加载实验报告

## 1. 系统架构
本实验实现了一个自定义的 `ImageOCRReader`，将 OCR 技术集成到 LlamaIndex 的数据加载流程中。

*   **输入**: 图像文件 (png, jpg)
*   **核心模块**: 
    *   `PaddleOCR`: 负责检测和识别图像中的文字。
    *   `ImageOCRReader`: 继承自 `BaseReader`，负责调度 OCR 并将结果封装为 `Document` 对象。
*   **输出**: LlamaIndex `Document` 对象（包含识别出的文本和元数据）。

## 2. 核心代码说明
`ImageOCRReader` 的关键逻辑如下：
1.  **依赖修复**: 由于新版 `paddlex` 与 `langchain` 存在兼容性问题，我们在代码头部使用了 **Monkey Patch** 技术模拟了 `langchain.docstore` 等模块。
2.  **初始化**: 加载 PaddleOCR 模型，显式禁用了方向分类等功能以适应新版 API。
3.  **load_data**: 
    *   遍历图像文件列表。
    *   调用 `self.ocr.predict(path)` 获取识别结果。
    *   解析 `OCRResult` 对象（通过访问 `rec_texts` 字段）提取文本列表。
    *   将识别出的文本行拼接成一个字符串。
    *   创建 `Document` 对象，并将文件路径等信息存入 `metadata`。

## 3. 实验结果

### 识别效果示例

| 图像内容 | 识别出的文本 (Key Findings) | RAG 问答效果 |
| :--- | :--- | :--- |
| **Invoice (表格)** | "Invoice #12345", "Banana", "$0.80" | **成功**：准确回答 "The price of Banana is $0.80." |
| **Meeting (列表)** | "Meeting Minutes", "Decisions: 1. Launch date set to Nov 15." | (预期成功)：能够提取关键日期和决策。 |
| **Sign (警告)** | "Warning: High Voltage!", "Do not touch." | (预期成功)：能够识别出警告内容。 |

### 局限性分析
1.  **空间结构丢失**: OCR 主要是按行识别，对于复杂的表格结构，简单的文本拼接可能会丢失行列对应关系。虽然本例中发票结构简单被正确识别，但复杂的跨行表格可能难以处理。
2.  **API 兼容性**: PaddleOCR 新旧版本 API 差异巨大（如 `use_gpu` 参数移除、`predict` 返回结构变化），集成时需要做大量适配。

## 4. 改进思路
*   **Structure Analysis**: 使用 PP-Structure 等工具进行版面分析，专门处理表格和段落，保留文档结构。
*   **多模态大模型**: 直接使用 GPT-4o 或 Qwen-VL 等多模态模型来理解图像，而不是通过 OCR 转文本，可能对复杂场景理解更好。
