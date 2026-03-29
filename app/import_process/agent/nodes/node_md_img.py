import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt



"""
  主要目标： 将md中图片进行单独处理，方便后去模型识别图片的含义！
  主要动作： 图片->文件服务器-> 图片网络地址    （上文100）图片（下文100）->视觉模型-> 图片总结  
           ---》 [图片的总结](网络图片地址) -> state ->  md_content == 新的内容（图片处理后的）|| md_path = 处理后的md的地址
  总结技术：
        minio
        视觉模型： 提示词 + 访问 
  总结步骤： 
     1. 校验并且获取本次操作的数据 
        参数： state  -> md_path md_content 
        响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
     2. 识别md中使用过的图片，采取做下一步（进行图片总结）
        参数： 1. md_content 2. images图片的文件夹地址
        响应： [(图片名,图片地址,(上文,下文))]
     3. 进行图片内容的总结和处理 （视觉模型）
        参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应： {图片名:总结,......}
     4. 上传图片minio以及更新md的内容 
        参数：minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文))] (minio) || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应：new_md_content
        state[md_content] = new_md_content
     5. 进行数据的最终处理和备份 
        参数：new_md_content , 原md地址 -》 xx.md -> xx_new.md  
        响应：新的md的地址 new_md_path 
        state[md_path] = new_new_md_path
    return state
"""

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    这部分定义了一个白名单（.jpg, .png 等）。当侦探看到一个文件时，先查验它到底是不是真正的图片。
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    检查文件到底在不在，把 Markdown 的文本内容读到内存里，并且定位存放图片的 images 文件夹在哪
    :param state:
    :return:
    """
    # 1. 获取md的地址 md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为空！")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path} 文件不存在！")

    # 要读取md_content
    if not state['md_content']:
        # 没有，再读取！ 有，证明是pdf节点解析过来的，已经给md_content进行赋值了！
        with md_path_obj.open("r", encoding="utf-8") as f:
            # 将所有文字内容打包成一个大字符串赋值给md_content
            md_content = f.read()
        state['md_content'] = md_content
        """
        关于with的拆解步骤
        f = md_path_obj.open("r")  # 1. 找管理员借出古籍
        md_content = f.read()      # 2. 阅读并抄写里面的内容
        f.close()                  # 3. 必须手动把古籍还给管理员！
        """

    # 图片文件夹obj
    # 注意：自己传入的md -》 你的图片文件夹也必须交 images
    images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file, context_length: int = 100):
    """
      拿着图片的名字，去茫茫的 Markdown 文本里用正则表达式搜索
      找到这个图片被插入的具体位置，并向前、向后各截取 100 个字符
    从md_content识别图片的上下文！
    约定上下文长度100
    :param md_content:
    :param image_file: cat.png
    :param context_length：默认截取长度
    :return:result：[(上文),(下文)]
    """
    """
    # 你好啊
    我很好，还有7行代码今天就结束了！小伙伴们坚持好！谢谢！
    哈哈
    哈
    嘿嘿
    【start】 ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)【end】
    啦啦啦啦
    巴巴爸爸
    ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
    嘿嘿额
    file_name zhaoweifeng.jpg
    """
    # 定义正则表达式  .*  .*?
    # r -> 不要把它们当成普通的换行符或转义符给吞了
    # 给个模版（通缉令）
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")

    results = []  # 存储图片多处使用，上下文不同 ！ 本次暴力处理，获取第一个！
    # 查询符合位置
    # 拿着模版去md_content中找出现的位置
    # 注意：finditer不会找到一个就停下，此处写死只要第一个
    # pattern.finditer->执行结果： 它找到了两个目标，排成了队伍交给 for 循环：
         # 1号目标：![大猫](cat.png)
         # 2号目标：![小猫](cat.png)
    for item in pattern.finditer(md_content):
        start, end = item.span()  # span获取匹配对象的起始和终止的位置
        # 截取上文
        pre_text = md_content[max(start - context_length, 0):start]  # 考虑前面有没有context_length 没有从0开始
        post_text = md_content[end:min(end + context_length, len(md_content))]  # 考虑后面有没有context_length 没有就到长度
        # 截取下文
        # results->列表变成了：[("我看到", "在睡觉。")]
        results.append((pre_text, post_text))
    # 截取位置前后的内容
    if results:
        logger.info(f"图片：{image_file} ,在{md_content[:100]}中使用了：{len(results)}次，截取第一个上下文：{results[0]}")
        # 返回第一个上下文[("我看到", "在睡觉。")]
        return results[0]


def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    进行md中图片识别，并且截取图片对应的上下文环境
    把 images 文件夹里的图片一张张拿出来，调用上面的勘探方法。
    最后生成一份清单（包含图片名、路径、上下文）
    :param md_content:
    :param images_dir_obj:
    :return: [(图片名，图片地址，上下元组())]
    """
    # 1. 我们先创建一个目标集合
    targets = []
    # 2. 循环读取images中的所有图片，校验在md中是否使用，使用了就截取上下文
    # 列出指定文件夹里面所有的文件和子文件夹的名称。
    # os.listdir()->['cat.png', 'dog.jpg', 'readme.txt', '新文件夹']
    for image_file in os.listdir(images_dir_obj):
        # 遍历每个文件的名字
        # 检查图片是否可用 -》 图片
        if not is_supported_image(image_file):
            logger.warning(f"当前文件：{image_file},不是图片格式，无需处理！")
            continue
        # 是图片，我们就在md查询，看是否存在，存在，读取对应的上下文即可
        # （上，下文）
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f"图片：{image_file}没有在md内容使用！上下文为空！")
            continue
                        # [(图片名，图片地址，上下文元组(上文，下文))]
                                      # Path("D:/项目/images") / "cat.png" -> Path("D:/项目/images/cat.png")
        targets.append((image_file, str(images_dir_obj / image_file), content_data))

    return targets


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    # 1. 校验并且获取本次操作的数据
    #         参数： state  -> md_path md_content
    #         响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 如果没有图片，则直接返回 state
    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}]没有图片，直接返回 state ！")
        return state
    # 2. 识别md中使用过的图片，采取做下一步（进行图片总结）
    # [(图片名,图片地址,(上文,下文 = 100))，(图片名,图片地址,(上文,下文 = 100))，(图片名,图片地址,(上文,下文 = 100))]
    targets = step_2_scan_images(md_content, images_dir_obj)
    #         参数： 1. md_content 2. images图片的文件夹地址
    #         响应： [(图片名,图片地址,(上文,下文))]
    return state