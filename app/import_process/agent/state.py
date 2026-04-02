from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如state["session_id"]、state.get("embedding_chunks")）
    """
    # 任务唯一ID，用于追踪日志
    task_id: str

    # --- 流程控制标记 ---
    is_md_read_enabled: bool   # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径

    # --- 切块相关 --- 【没用】
    # is_normal_split_enabled: bool
    # is_silicon_flow_api_enabled: bool
    # is_advanced_split_enabled: bool
    # is_vllm_enabled: bool

    # --- 路径相关 ---
    local_dir: str        # 当前工作目录或输出目录
    local_file_path: str  # 用户最初上传的原始文件所在的路径
    file_title: str       # 去掉后缀名的纯文件名（比如原始文件叫“使用说明.pdf”，这里就是“使用说明”）
    pdf_path: str         # PDF 文件路径 (如果输入是PDF)，方便后续节点去磁盘上读取物理文件
    md_path: str          # Markdown 文件路径 (转换后或直接输入的)，方便后续节点去磁盘上读取物理文件
    # split_path: str       # 分块后的文件路径 【没用】
    # embeddings_path: str  # 向量数据库文件路径【没用】

    # --- 内容数据 ---
    md_content: str       # 存放整篇文档转换出来的 Markdown 纯文本。这是所有文字处理的基础。
    chunks: list          # 一个列表，存放文档被“切肉师傅”（切分节点）切成的一段段小文本（切片）。
                           # 每个小段落不仅有文字，还带着它属于哪个大标题等“标签”（metadata）
    item_name: str        # 识别出的核心商品或主体名称（例如“华为路由器AX3”）
                            # 通俗理解：这是为了给所有的内容打上“防伪溯源标签”。
                            # 大模型识别出商品名后放在这里，之后生成的每一个小切片（chunk）都会贴上这个 item_name，
                            # 防止将来用户提问时，把“路由器”的说明书和“微波炉”的说明书搞混

    # --- 数据库相关 ---
    embeddings_content: list # 一个列表，存放已经被转化为计算机能理解的“向量”（数字矩阵）的数据。
                             # 当数据来到这个格子里时，意味着它已经彻底加工完毕，随时准备被一脚踢进 Milvus 向量数据库里永久保存了


# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    # "is_normal_split_enabled": True,
    # "is_silicon_flow_api_enabled": True,
    # "is_advanced_split_enabled": False,
    # "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    # "split_path": "",
    # "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    # 当有一个新文档要进入流水线时，系统就会调用这个函数来准备初始收纳盒！！！！！！！！
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）
    Returns:
        新的状态实例
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    #为什么要用深拷贝（Deep Copy）？
    #灾难场景（不使用深拷贝）：假设你不用深拷贝，而是直接修改 graph_default_state。这就好比医院里所有人共用同一份体检表原件。文档 A 把切分好的段落塞进了 chunks 列表，紧接着文档 B 进来时，会发现 chunks 里竟然还残留着文档 A 的内容（这在并发编程中叫“全局状态污染”）。
    #深拷贝的作用：每次来一个新任务，系统都会去仓库里拿那个标准原型盒，从里到外全新克隆出一个完全独立的物理盒子。这样无论你怎么在这个新盒子里折腾列表或字典，都不会影响到原型的干净状态，也不会干扰其他同时运行的任务。
    #覆盖参数 (**overrides)：系统在把这个全新克隆的盒子放上流水线前，顺手把外界传入的已知信息（比如你测试代码里传的 local_file_path="万用表RS-12的使用.pdf"）直接放进对应的格子里。
    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)