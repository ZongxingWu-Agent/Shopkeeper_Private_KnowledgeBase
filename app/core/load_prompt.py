from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger  # 可选，加日志更友好

def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词并渲染变量占位符
    :param name: 提示词文件名（不带.prompt后缀，如image_summary）
    :param **kwargs: 需渲染的变量键值对（如root_folder="测试文件", image_content=("上文内容", "下文内容")）
    :return: 渲染后的最终提示词字符串
    """

    """
    函数签名里的 **kwargs 是 Python 里非常强大的“万能接收器”（关键字可变参数）。
    当你这样调用时：
    load_prompt(name='image_summary', root_folder="说明书", image_content=("上", "下"))
    **kwargs 会自动把 name 后面所有自带名字的参数，打包成一个字典收好：
    {'root_folder': '说明书', 'image_content': ('上', '下')}
    这就像是顾客给奶茶机输入了一长串不固定的加料配方。
    """

    # 1. 拼接提示词路径（你的原有逻辑，完全保留）
    # 代码先去 prompts 文件夹下找到对应的 .prompt 文件。如果找不到，立马报错提醒（防止程序盲目运行）
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'

    # 2. 校验文件是否存在（可选，避免文件不存在直接报错）
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")

    # 3. 读取纯文本提示词（你的原有逻辑）
    # 如果找到了，就用 read_text 把文件里的内容原封不动地读出来，赋值给 raw_prompt
    # raw_prompt->template
    raw_prompt = prompt_path.read_text(encoding='utf-8')

    # 4. 核心：如果传了参数，渲染占位符；没传参，直接返回原文本!!!!!!!!!!!!
    if kwargs:
        # 这里的 **kwargs 会把刚才那个字典“拆开”，把里面的值精准地填进 raw_prompt 文本中对应名字的 {} 坑位里
        # rendered_prompt->prompt
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        # 一瞬间，带着大括号的半成品模板，就被渲染成了可以直接端给大模型阅读的通顺大白话 rendered_prompt
        return rendered_prompt
    # 5. 没传参，直接返回原文本.提示词有可能不需要传参？？？？？？？？
    return raw_prompt



if __name__ == '__main__':
    # 测试：传入参数渲染占位符（和业务代码中实际使用方式一致）
    root_folder = "hl3070使用说明书"  # 要替换的文件名称
    image_content = ("这是图片的上文内容", "这是图片的下文内容")  # 要替换的上下文
    # 调用时传入所有需要渲染的变量（键名必须和.prompt中的占位符完全一致）
    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,  # 对应{root_folder}
        image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
    )
    print("✅ 渲染后的最终提示词：")
    print(final_prompt)