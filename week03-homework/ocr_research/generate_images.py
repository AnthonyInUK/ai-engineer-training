from PIL import Image, ImageDraw, ImageFont
import os


def create_text_image(filename, text, size=(800, 600), bg_color='white', text_color='black'):
    img = Image.new('RGB', size, color=bg_color)
    d = ImageDraw.Draw(img)

    # 尝试加载默认字体，如果失败则使用默认位图字体
    try:
        # Mac OS 常用字体路径
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
    except:
        try:
            # Ubuntu 常用字体
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            # 兜底
            font = ImageFont.load_default()

    # 简单的文本换行逻辑
    margin = 50
    offset = 50
    for line in text.split('\n'):
        d.text((margin, offset), line, fill=text_color, font=font)
        offset += 60

    img.save(filename)
    print(f"Created {filename}")


def main():
    if not os.path.exists("test_images"):
        os.makedirs("test_images")

    text1 = """
    Invoice #12345
    Date: 2023-10-27
    
    Item          Price
    -------------------
    Apple         $1.50
    Banana        $0.80
    Orange        $1.20
    
    Total:        $3.50
    """
    create_text_image("test_images/invoice.png", text1)

    text2 = """
    Meeting Minutes
    Topic: Project Alpha
    Date: 2023-10-28
    
    Attendees: Alice, Bob, Charlie
    
    Decisions:
    1. Launch date set to Nov 15.
    2. Budget approved for Q4.
    """
    create_text_image("test_images/meeting.png", text2)

    text3 = """
    Warning: High Voltage!
    Do not touch.
    Authorized personnel only.
    """
    create_text_image("test_images/sign.png", text3,
                      bg_color='yellow', text_color='red')


if __name__ == "__main__":
    main()
